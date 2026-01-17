# -*- coding: utf-8 -*-
import os, re, json, hashlib
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo

import requests
import dateparser

BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip()
SOURCE_ID = os.getenv("TG_SOURCE_CHAT_ID", "").strip()   # где бот читает события (группа/канал)
TARGET_ID = os.getenv("TG_TARGET_CHAT_ID", "").strip()   # куда слать напоминания
CAL_ID    = os.getenv("TG_CALENDAR_CHAT_ID", "").strip() # где держать закреп "Календарь" (обычно TARGET)
TZ_NAME   = os.getenv("TZ", "Europe/Berlin").strip()
ALLOWED_THREAD_ID = os.getenv("TG_ALLOWED_THREAD_ID", "").strip()  # опционально

if not BOT_TOKEN:
    raise SystemExit("TG_BOT_TOKEN не задан")

API = f"https://api.telegram.org/bot{BOT_TOKEN}"
TZ = ZoneInfo(TZ_NAME)

def tg(method, payload=None, timeout=30):
    payload = payload or {}
    r = requests.post(f"{API}/{method}", json=payload, timeout=timeout)
    data = r.json()
    if not data.get("ok"):
        return None, data
    return data["result"], data

def now():
    return datetime.now(TZ)

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

STATE_MARKER = "#STATE_JSON"
CALENDAR_TITLE = "Календарь (Катя / Женя)"
MAX_MESSAGE_LEN = 3900
TOLERANCE_MINUTES = 7

DEFAULT_CFG = {
    "tz": TZ_NAME,
    "require_bang": True,
    "react_ok": True,
    "reminder_times": {"pre": "08:00", "dayof": "08:00"},
    "people": {
        "Катя": {"default_offsets": [7, 2, 1, 0], "default_loud": False},
        "Женя": {"default_offsets": [7, 2, 1, 0], "default_loud": False}
    },
    "recurring": {"enabled": False, "items": []}
}

def safe_json_load(s):
    try:
        return json.loads(s)
    except:
        return None

def extract_state_from_text(text: str):
    if not text or STATE_MARKER not in text:
        return None
    m = re.search(rf"{re.escape(STATE_MARKER)}\s*```json\s*(\{{.*?\}})\s*```", text, re.S)
    if not m:
        return None
    return safe_json_load(m.group(1))

def parse_hhmm(s: str):
    h, m = s.strip().split(":")
    return int(h), int(m)

def detect_has_time(s: str):
    return bool(re.search(r"\b([01]?\d|2[0-3])[:.,][0-5]\d\b", s))

def parse_datetime_from_text(s: str):
    return dateparser.parse(
        s,
        languages=["ru", "en", "de"],
        settings={
            "TIMEZONE": TZ_NAME,
            "RETURN_AS_TIMEZONE_AWARE": True,
            "PREFER_DATES_FROM": "future",
            "DATE_ORDER": "DMY"
        }
    )

NAME_SET = {"катя": "Катя", "женя": "Женя"}

def strip_prefix_bang(text: str, require_bang: bool):
    t = (text or "").strip()
    if require_bang:
        if not t.startswith("!"):
            return None
        t = t[1:].strip()
    else:
        if t.startswith("!"):
            t = t[1:].strip()
    return t

def parse_events_from_message(text: str, cfg: dict):
    t = strip_prefix_bang(text, cfg.get("require_bang", True))
    if t is None:
        return []

    lines = [x.strip() for x in t.splitlines() if x.strip()]
    events = []
    current_person = "Катя"

    for line in lines:
        low = line.lower().strip()
        if low in NAME_SET:
            current_person = NAME_SET[low]
            continue

        m = re.match(r"^(Катя|Женя)\b[:\- ]*(.*)$", line, re.I)
        if m:
            current_person = NAME_SET[m.group(1).lower()]
            line_rest = m.group(2).strip()
        else:
            line_rest = line

        dt = parse_datetime_from_text(line_rest)
        if not dt:
            continue

        has_time = detect_has_time(line_rest)
        dt = dt.astimezone(TZ)
        if not has_time:
            dt = dt.replace(hour=9, minute=0, second=0, microsecond=0)

        pconf = cfg.get("people", {}).get(current_person, {})
        offsets = pconf.get("default_offsets", [7, 2, 1, 0])
        loud = bool(pconf.get("default_loud", False))

        title = line_rest.strip()
        eid = sha1(f"{current_person}|{dt.isoformat()}|{title}")

        events.append({
            "id": eid,
            "person": current_person,
            "dt_iso": dt.isoformat(),
            "has_time": has_time,
            "title": title,
            "offsets": offsets,
            "loud": loud
        })
    return events

def build_calendar_text(state: dict):
    cfg = state.get("cfg", DEFAULT_CFG)
    events = state.get("events", [])

    by_person = {"Катя": [], "Женя": []}
    for ev in events:
        by_person.setdefault(ev.get("person","Катя"), []).append(ev)

    for p in by_person:
        by_person[p].sort(key=lambda ev: ev.get("dt_iso",""))

    lines = []
    lines.append(CALENDAR_TITLE)
    lines.append("")
    lines.append(f"Часовой пояс: {cfg.get('tz', TZ_NAME)}")
    lines.append("")

    for p in ["Катя", "Женя"]:
        lines.append(p)
        if not by_person.get(p):
            lines.append("• (пока нет будущих событий)")
        else:
            for ev in by_person[p]:
                dt = datetime.fromisoformat(ev["dt_iso"]).astimezone(TZ)
                dt_str = dt.strftime("%d.%m.%Y %H:%M") if ev.get("has_time", True) else dt.strftime("%d.%m.%Y")
                title = ev.get("title","").strip() or "(без названия)"
                offs = ev.get("offsets", [])
                offs_str = ",".join([str(x) for x in offs]) if offs else "-"
                loud = "звук" if ev.get("loud", False) else "тихо"
                lines.append(f"• {dt_str} — {title}  | напом.: {offs_str} | {loud}")
        lines.append("")

    slim = {
        "v": 1,
        "cfg": cfg,
        "calendar_message_id": state.get("calendar_message_id"),
        "last_update_id": state.get("last_update_id", 0),
        "events": events,
        "sent": state.get("sent", {}),
        "sent_recurring": state.get("sent_recurring", {})
    }
    state_json = json.dumps(slim, ensure_ascii=False, separators=(",", ":"))
    lines.append(STATE_MARKER)
    lines.append("```json")
    lines.append(state_json)
    lines.append("```")

    text = "\n".join(lines)
    return text[:MAX_MESSAGE_LEN]

def get_pinned_calendar(chat_id: str):
    res, _ = tg("getChat", {"chat_id": chat_id})
    if not res:
        return None
    return res.get("pinned_message")

def ensure_calendar(state: dict, cal_chat_id: str):
    pinned = get_pinned_calendar(cal_chat_id)
    if pinned:
        st = extract_state_from_text(pinned.get("text",""))
        if st:
            state.update(st)
            state["calendar_message_id"] = pinned["message_id"]
            return state

    base_state = {
        "cfg": DEFAULT_CFG,
        "events": [],
        "sent": {},
        "sent_recurring": {},
        "last_update_id": state.get("last_update_id", 0),
        "calendar_message_id": None
    }
    text = build_calendar_text(base_state)
    msg, _ = tg("sendMessage", {"chat_id": cal_chat_id, "text": text, "disable_web_page_preview": True})
    if not msg:
        return state
    mid = msg["message_id"]
    tg("pinChatMessage", {"chat_id": cal_chat_id, "message_id": mid, "disable_notification": True})
    base_state["calendar_message_id"] = mid
    return base_state

def update_calendar_message(state: dict, cal_chat_id: str):
    mid = state.get("calendar_message_id")
    if not mid:
        return
    text = build_calendar_text(state)
    tg("editMessageText", {"chat_id": cal_chat_id, "message_id": mid, "text": text, "disable_web_page_preview": True})

def react_ok(chat_id: str, message_id: int, cfg: dict):
    if not cfg.get("react_ok", True):
        return
    res, _ = tg("setMessageReaction", {
        "chat_id": chat_id,
        "message_id": message_id,
        "reaction": [{"type": "emoji", "emoji": "✅"}]
    })
    if res is None:
        tg("sendMessage", {"chat_id": chat_id, "text": "Добавлено ✅", "reply_to_message_id": message_id})

def due_for_offset(ev_dt: datetime, offset_days: int, cfg: dict, now_dt: datetime):
    times = cfg.get("reminder_times", DEFAULT_CFG["reminder_times"])
    hhmm = times["dayof"] if offset_days == 0 else times["pre"]
    hh, mm = parse_hhmm(hhmm)
    target_date = (ev_dt.date() - timedelta(days=offset_days))
    target_dt = datetime.combine(target_date, dtime(hh, mm), TZ)
    delta = now_dt - target_dt
    return timedelta(minutes=0) <= delta <= timedelta(minutes=TOLERANCE_MINUTES)

def send_event_reminder(ev: dict, offset: int, target_chat_id: str):
    ev_dt = datetime.fromisoformat(ev["dt_iso"]).astimezone(TZ)
    person = ev.get("person","")
    title = ev.get("title","")
    when = ev_dt.strftime("%d.%m.%Y %H:%M") if ev.get("has_time", True) else ev_dt.strftime("%d.%m.%Y")
    prefix = f"{person}: " if person else ""
    text = f"{prefix}Сегодня: {when} — {title}" if offset == 0 else f"{prefix}Через {offset} дн.: {when} — {title}"
    tg("sendMessage", {"chat_id": target_chat_id, "text": text, "disable_notification": (not ev.get("loud", False))})

def cleanup_events(state: dict):
    n = now()
    keep = []
    for ev in state.get("events", []):
        dt = datetime.fromisoformat(ev["dt_iso"]).astimezone(TZ)
        if dt < n - timedelta(days=1):
            continue
        keep.append(ev)
    state["events"] = keep

    existing = {ev["id"] for ev in keep}
    sent = state.get("sent", {})
    state["sent"] = {k: v for k, v in sent.items() if k.split("|",1)[0] in existing}

def apply_cfg_from_text(text: str):
    m = re.search(r"/setcfg\b(.*)$", text, re.S)
    if not m:
        return None
    tail = m.group(1).strip()
    if not tail:
        return None
    m2 = re.search(r"```json\s*(\{.*\})\s*```", tail, re.S)
    if m2:
        tail = m2.group(1).strip()
    return safe_json_load(tail)

def allowed_thread_ok(msg: dict) -> bool:
    if not ALLOWED_THREAD_ID:
        return True
    tid = msg.get("message_thread_id")
    return str(tid) == str(ALLOWED_THREAD_ID)

def main():
    # Определяем где читаем, куда шлём, где календарь
    source_chat_id = SOURCE_ID or TARGET_ID
    target_chat_id = TARGET_ID or SOURCE_ID
    cal_chat_id = CAL_ID or target_chat_id or source_chat_id

    if not source_chat_id:
        raise SystemExit("Нужно задать TG_SOURCE_CHAT_ID или TG_TARGET_CHAT_ID")

    state = {"cfg": DEFAULT_CFG, "events": [], "sent": {}, "sent_recurring": {}, "last_update_id": 0, "calendar_message_id": None}

    # Пытаемся подхватить state из закрепа (если календарь уже создан)
    pinned = get_pinned_calendar(cal_chat_id)
    if pinned:
        st = extract_state_from_text(pinned.get("text",""))
        if st:
            state.update(st)
            state["calendar_message_id"] = pinned["message_id"]

    cfg = state.get("cfg", DEFAULT_CFG)

    offset = int(state.get("last_update_id", 0)) + 1
    updates, _ = tg("getUpdates", {"offset": offset, "timeout": 0, "limit": 100})
    if updates is None:
        updates = []
    max_uid = state.get("last_update_id", 0)

    for upd in updates:
        uid = upd.get("update_id", 0)
        max_uid = max(max_uid, uid)

        msg = upd.get("message") or upd.get("edited_message")
        if not msg:
            msg = upd.get("channel_post") or upd.get("edited_channel_post")
        if not msg:
            continue

        chat = msg.get("chat", {})
        chat_id = str(chat.get("id"))
        text = (msg.get("text") or "").strip()
        mid = msg.get("message_id")

        # Команды обслуживаем только в календарном чате (чтобы не путаться)
        if chat_id == str(cal_chat_id):
            if text.startswith("/setup"):
                state = ensure_calendar(state, cal_chat_id)
                tg("sendMessage", {"chat_id": cal_chat_id, "text": "Готово: календарь создан и закреплён."})
                cfg = state.get("cfg", DEFAULT_CFG)
                continue

            if text.startswith("/getid"):
                tid = msg.get("message_thread_id")
                tg("sendMessage", {"chat_id": cal_chat_id, "text": f"chat_id={chat_id}\nthread_id={tid}"})
                continue

            if text.startswith("/threadid"):
                tid = msg.get("message_thread_id")
                tg("sendMessage", {"chat_id": cal_chat_id, "text": f"thread_id={tid}"})
                continue

            if text.startswith("/setcfg"):
                new_cfg = apply_cfg_from_text(text)
                if new_cfg and isinstance(new_cfg, dict):
                    merged = DEFAULT_CFG.copy()
                    merged.update(new_cfg)
                    merged["people"] = DEFAULT_CFG["people"].copy()
                    merged["people"].update(new_cfg.get("people", {}))
                    merged["reminder_times"] = DEFAULT_CFG["reminder_times"].copy()
                    merged["reminder_times"].update(new_cfg.get("reminder_times", {}))
                    merged["recurring"] = DEFAULT_CFG["recurring"].copy()
                    merged["recurring"].update(new_cfg.get("recurring", {}))
                    state["cfg"] = merged
                    cfg = merged
                    tg("sendMessage", {"chat_id": cal_chat_id, "text": "Настройки применены."})
                    update_calendar_message(state, cal_chat_id)
                else:
                    tg("sendMessage", {"chat_id": cal_chat_id, "text": "Не смог прочитать JSON. Формат: /setcfg {\"require_bang\":true,...}"})
                continue

        # События читаем только из source_chat_id
        if chat_id != str(source_chat_id):
            continue

        # если это тема, и задан thread filter
        if not allowed_thread_ok(msg):
            continue

        # если календарь ещё не создан — ждём /setup
        if not state.get("calendar_message_id"):
            continue

        new_events = parse_events_from_message(text, cfg)
        if new_events:
            existing = {ev["id"] for ev in state.get("events", [])}
            added = 0
            for ev in new_events:
                if ev["id"] not in existing:
                    state["events"].append(ev)
                    added += 1
            if added > 0:
                react_ok(source_chat_id, mid, cfg)

    state["last_update_id"] = max_uid

    if state.get("calendar_message_id"):
        cleanup_events(state)

        # напоминания
        n = now()
        sent = state.get("sent", {})
        for ev in state.get("events", []):
            ev_dt = datetime.fromisoformat(ev["dt_iso"]).astimezone(TZ)
            for off in ev.get("offsets", []):
                key = f"{ev['id']}|{int(off)}"
                if sent.get(key):
                    continue
                if due_for_offset(ev_dt, int(off), cfg, n):
                    send_event_reminder(ev, int(off), target_chat_id or cal_chat_id)
                    sent[key] = n.isoformat()
        state["sent"] = sent

        update_calendar_message(state, cal_chat_id)

if __name__ == "__main__":
    main()

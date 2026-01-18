# -*- coding: utf-8 -*-
import os, re, json, hashlib
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
import requests
import dateparser

BOT_TOKEN = (os.getenv("TG_BOT_TOKEN") or "").strip()
TZ_NAME = (os.getenv("TZ") or "Europe/Berlin").strip()

SOURCE_CHAT_ID = (os.getenv("SOURCE_CHAT_ID") or "").strip()
SOURCE_THREAD_ID = (os.getenv("SOURCE_THREAD_ID") or "").strip()

STATE_CHAT_ID = (os.getenv("STATE_CHAT_ID") or "").strip()
STATE_THREAD_ID = (os.getenv("STATE_THREAD_ID") or "").strip()  # опционально: тема в storage-группе

USER1_CHAT_ID = (os.getenv("USER1_CHAT_ID") or "").strip()
USER2_CHAT_ID = (os.getenv("USER2_CHAT_ID") or "").strip()

PEOPLE_MARKERS_JSON = (os.getenv("PEOPLE_MARKERS_JSON") or "").strip()

if not BOT_TOKEN:
    raise SystemExit("TG_BOT_TOKEN is required")

API = f"https://api.telegram.org/bot{BOT_TOKEN}"
TZ = ZoneInfo(TZ_NAME)

STATE_MARKER = "#STATE_JSON"
MAXLEN = 3800
TOL_MIN = 7

DEFAULT_CFG = {
    "reminder_times": {"pre": "08:00", "dayof": "08:00"},  # за день и в день — утром
    "offsets_days": [1, 0],
    "silent_default": True,       # без звука по умолчанию
    "daily_digest_enabled": True, # ежедневный список "на сегодня"
    "daily_digest_time": "08:05",
}

def tg(method, payload=None, timeout=30):
    payload = payload or {}
    r = requests.post(f"{API}/{method}", json=payload, timeout=timeout)
    data = r.json()
    if not data.get("ok"):
        return None, data
    return data["result"], data

def now():
    return datetime.now(TZ)

def hhmm(s: str):
    h, m = s.strip().split(":")
    return int(h), int(m)

def safe_json(s):
    try:
        return json.loads(s)
    except:
        return None

def extract_state(text: str):
    if not text or STATE_MARKER not in text:
        return None
    m = re.search(rf"{re.escape(STATE_MARKER)}\s*```json\s*(\{{.*?\}})\s*```", text, re.S)
    if not m:
        return None
    return safe_json(m.group(1))

def build_state_text(state: dict):
    lines = [
        "STATE (служебное сообщение бота). Не удалять.",
        STATE_MARKER,
        "```json",
        json.dumps(state, ensure_ascii=False, separators=(",", ":")),
        "```"
    ]
    return ("\n".join(lines))[:MAXLEN]

def parse_dt(text: str):
    return dateparser.parse(
        text,
        languages=["ru", "en", "de"],
        settings={
            "TIMEZONE": TZ_NAME,
            "RETURN_AS_TIMEZONE_AWARE": True,
            "PREFER_DATES_FROM": "future",
            "DATE_ORDER": "DMY",
        }
    )

def detect_time(s: str):
    return bool(re.search(r"\b([01]?\d|2[0-3])[:][0-5]\d\b", s))

def normalize_time_in_line(line: str) -> str:
    # 1) "в 8.10" / "в 8,10" -> "в 8:10"
    line = re.sub(r"\bв\s*([01]?\d|2[0-3])[.,]([0-5]\d)\b", r"в \1:\2", line, flags=re.I)
    # 2) "23 января 9,40" -> "23 января 9:40"
    months = r"(январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)"
    line = re.sub(rf"(\b\d{{1,2}}\s+{months}\w*\b.*?)(\b([01]?\d|2[0-3])[.,]([0-5]\d)\b)",
                  lambda m: m.group(1) + m.group(3) + ":" + m.group(4),
                  line, flags=re.I)
    # 3) "09.03.26 8.10" (без "в") -> "09.03.26 8:10"
    line = re.sub(r"(\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b)\s+([01]?\d|2[0-3])[.,]([0-5]\d)\b", r"\1 \2:\3", line)
    return line

def load_markers():
    """
    PEOPLE_MARKERS_JSON лежит только в Secrets.
    Пример значения:
      {"1":["Женя"],"2":["Катя"]}
    Имена в код не попадают.
    """
    if not PEOPLE_MARKERS_JSON:
        return {"1": [], "2": []}
    obj = safe_json(PEOPLE_MARKERS_JSON)
    if not isinstance(obj, dict):
        return {"1": [], "2": []}
    out = {"1": [], "2": []}
    for k in ("1", "2"):
        v = obj.get(k, [])
        if isinstance(v, list):
            out[k] = [str(x).strip() for x in v if str(x).strip()]
    return out

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def is_category_line(line: str) -> bool:
    # категория: нет цифр, не слишком длинная
    if any(ch.isdigit() for ch in line):
        return False
    if len(line) > 40:
        return False
    # не служебные слова
    if line.strip() in ("-", "—"):
        return False
    return True

def parse_events_from_block(text: str, markers: dict):
    """
    Парсит текст как на твоём скрине:
      Заголовок
      <маркер списка 1>
      даты...
      <категория>
      даты...
      -
      <маркер списка 2>
      даты...
    """
    if not text:
        return []

    # команды пропускаем
    if text.strip().startswith("/"):
        return []

    slot = "1"
    category = None
    events = []

    # нормализуем переносы
    lines = [x.strip() for x in text.splitlines() if x.strip()]

    # быстрые множества маркеров
    m1 = set([x.casefold() for x in markers.get("1", [])])
    m2 = set([x.casefold() for x in markers.get("2", [])])

    for raw in lines:
        line = raw.strip()

        # разделители
        if line in ("-", "—"):
            category = None
            continue

        # переключение списка по маркерам (лежат в secrets)
        lf = line.casefold()
        if lf in m1:
            slot = "1"
            category = None
            continue
        if lf in m2:
            slot = "2"
            category = None
            continue

        # "шапки" типа "ТЕРМИНЫ" — игнорим
        if line.isupper() and len(line) <= 30:
            continue

        # пробуем распарсить дату/время
        norm = normalize_time_in_line(line)
        dt = parse_dt(norm)

        if dt:
            dt = dt.astimezone(TZ)
            has_time = detect_time(norm)
            if not has_time:
                dt = dt.replace(hour=9, minute=0, second=0, microsecond=0)

            title = line
            # если есть категория — добавим её к заголовку (без имён)
            if category:
                title = f"{category}: {line}"

            eid = sha1(f"{slot}|{dt.isoformat()}|{title}")
            events.append({
                "id": eid,
                "slot": slot,
                "dt_iso": dt.isoformat(),
                "has_time": has_time,
                "title": title
            })
            continue

        # если дата не распознана — возможно это категория
        if is_category_line(line):
            category = line
            continue

    return events

def send(chat_id: int, text: str, silent: bool = True, reply_markup=None):
    payload = {"chat_id": chat_id, "text": text[:MAXLEN], "disable_notification": silent}
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup
    return tg("sendMessage", payload)[0]

def answer_callback(cq_id: str, text: str):
    tg("answerCallbackQuery", {"callback_query_id": cq_id, "text": text, "show_alert": False})

def set_reaction_ok(chat_id: str, message_id: int):
    tg("setMessageReaction", {
        "chat_id": chat_id,
        "message_id": message_id,
        "reaction": [{"type": "emoji", "emoji": "✅"}]
    })

def due_at(target_dt: datetime, n: datetime):
    delta = n - target_dt
    return timedelta(minutes=0) <= delta <= timedelta(minutes=TOL_MIN)

def due_for_offset(ev_dt: datetime, offset_days: int, cfg: dict, n: datetime):
    times = cfg["reminder_times"]
    when = times["dayof"] if offset_days == 0 else times["pre"]
    h, m = hhmm(when)
    d = (ev_dt.date() - timedelta(days=offset_days))
    target = datetime.combine(d, dtime(h, m), TZ)
    return due_at(target, n)

def due_daily_digest(cfg: dict, n: datetime):
    if not cfg.get("daily_digest_enabled", True):
        return False
    h, m = hhmm(cfg.get("daily_digest_time", "08:05"))
    target = datetime.combine(n.date(), dtime(h, m), TZ)
    return due_at(target, n)

def format_event(ev: dict):
    dt = datetime.fromisoformat(ev["dt_iso"]).astimezone(TZ)
    ds = dt.strftime("%d.%m.%Y %H:%M") if ev.get("has_time") else dt.strftime("%d.%m.%Y")
    return f"• {ds} — {ev.get('title','')}".strip()

def build_all_events_text(state: dict):
    evs = list(state["events"].values())
    evs.sort(key=lambda x: x["dt_iso"])
    if not evs:
        return "Будущих событий нет."

    out = ["Все будущие запланированные события:", ""]
    for slot in ("1", "2"):
        items = [e for e in evs if e["slot"] == slot]
        out.append(f"Список {slot}:")
        if not items:
            out.append("• (нет)")
        else:
            for e in items:
                out.append(format_event(e))
        out.append("")
    return "\n".join(out)[:MAXLEN]

def get_allowed_ids():
    a = []
    if USER1_CHAT_ID.isdigit():
        a.append(int(USER1_CHAT_ID))
    if USER2_CHAT_ID.isdigit():
        a.append(int(USER2_CHAT_ID))
    return set(a)

def ensure_state():
    if not STATE_CHAT_ID:
        return None, None

    chat, _ = tg("getChat", {"chat_id": STATE_CHAT_ID})
    pinned = (chat or {}).get("pinned_message")
    if pinned:
        st = extract_state(pinned.get("text", ""))
        if st:
            return st, pinned["message_id"]

    base = {
        "v": 1,
        "tz": TZ_NAME,
        "cfg": DEFAULT_CFG,
        "last_update_id": 0,
        "events": {},    # id -> event
        "by_msg": {},    # source_msg_id -> [event_ids]
        "sent": {},      # keys for sent reminders
        "notes": ""      # /ideas
    }

    payload = {"chat_id": int(STATE_CHAT_ID), "text": build_state_text(base), "disable_notification": True}
    if STATE_THREAD_ID.isdigit():
        payload["message_thread_id"] = int(STATE_THREAD_ID)

    msg, _ = tg("sendMessage", payload)
    if not msg:
        raise SystemExit("Cannot create STATE message")

    mid = msg["message_id"]
    tg("pinChatMessage", {"chat_id": int(STATE_CHAT_ID), "message_id": mid, "disable_notification": True})
    return base, mid

def save_state(state_mid: int, state: dict):
    tg("editMessageText", {
        "chat_id": int(STATE_CHAT_ID),
        "message_id": state_mid,
        "text": build_state_text(state)
    })

def main():
    tg("deleteWebhook", {"drop_pending_updates": True})

    markers = load_markers()
    allowed_private = get_allowed_ids()

    state, state_mid = ensure_state()
    last = 0 if not state else int(state.get("last_update_id", 0))

    updates, _ = tg("getUpdates", {"offset": last + 1, "timeout": 0, "limit": 100})
    if updates is None:
        updates = []

    max_uid = last
    n = now()

    # 1) callback кнопок
    for upd in updates:
        max_uid = max(max_uid, upd.get("update_id", 0))
        cq = upd.get("callback_query")
        if not cq:
            continue
        data = cq.get("data")
        from_id = int(cq.get("from", {}).get("id", 0))
        if data == "GET_MY_ID":
            answer_callback(cq["id"], "Ок")
            send(from_id, f"Ваш chat_id: {from_id}", silent=True)
            continue

        if data == "SHOW_ALL":
            # если allowed ids уже заданы — ограничиваем
            if allowed_private and from_id not in allowed_private:
                answer_callback(cq["id"], "Нет доступа")
                continue
            answer_callback(cq["id"], "Ок")
            if state:
                send(from_id, build_all_events_text(state), silent=True)
            else:
                send(from_id, "STATE ещё не настроен (нет STATE_CHAT_ID).", silent=True)
            continue

    # 2) сообщения
    for upd in updates:
        msg = upd.get("message") or upd.get("edited_message")
        if not msg:
            continue

        chat = msg.get("chat", {})
        chat_type = chat.get("type")
        chat_id = int(chat.get("id"))
        text = (msg.get("text") or "").strip()
        mid = msg.get("message_id")
        thread_id = msg.get("message_thread_id")

        max_uid = max(max_uid, upd.get("update_id", 0))

        # команда /ids — чтобы узнавать chat_id и thread_id прямо в нужной теме
        if text.startswith("/ids"):
            info = f"chat_id={chat_id}\nthread_id={thread_id}"
            if thread_id:
                tg("sendMessage", {"chat_id": chat_id, "message_thread_id": int(thread_id), "text": info, "disable_notification": True})
            else:
                send(chat_id, info, silent=True)
            continue

        # личка: меню + кнопки
        if chat_type == "private":
            kb = {
                "inline_keyboard": [
                    [{"text": "Получить свой ID", "callback_data": "GET_MY_ID"}],
                    [{"text": "Показать запланированные события", "callback_data": "SHOW_ALL"}]
                ]
            }
            if text.startswith("/start") or text.startswith("/menu"):
                send(chat_id, "Меню:", silent=True, reply_markup=kb)
            elif text.startswith("/ideas"):
                if allowed_private and chat_id not in allowed_private:
                    continue
                if not state:
                    send(chat_id, "STATE ещё не настроен.", silent=True)
                else:
                    notes = (state.get("notes") or "").strip()
                    send(chat_id, notes if notes else "(идеи не заданы)", silent=True)
            elif text.startswith("/ideas_set"):
                if allowed_private and chat_id not in allowed_private:
                    continue
                if not state:
                    send(chat_id, "STATE ещё не настроен.", silent=True)
                else:
                    parts = text.split(maxsplit=1)
                    if len(parts) < 2:
                        send(chat_id, "Формат: /ideas_set текст", silent=True)
                    else:
                        state["notes"] = parts[1]
                        send(chat_id, "Ок.", silent=True)
            continue

        # источник: только заданная группа + заданная тема
        if state and SOURCE_CHAT_ID and SOURCE_THREAD_ID:
            if str(chat_id) == str(SOURCE_CHAT_ID) and str(thread_id) == str(SOURCE_THREAD_ID):
                if text.startswith("/"):
                    continue

                evs = parse_events_from_block(text, markers)
                if evs:
                    # заменить события, пришедшие из этого сообщения
                    old_ids = state["by_msg"].get(str(mid), [])
                    for eid in old_ids:
                        state["events"].pop(eid, None)

                    state["by_msg"][str(mid)] = [e["id"] for e in evs]
                    for e in evs:
                        state["events"][e["id"]] = e

                    set_reaction_ok(str(chat_id), mid)

    # 3) напоминания и чистка
    if state:
        state["last_update_id"] = max_uid

        # убрать сильно прошедшее
        keep = {}
        for eid, ev in state["events"].items():
            dt = datetime.fromisoformat(ev["dt_iso"]).astimezone(TZ)
            if dt >= n - timedelta(days=1):
                keep[eid] = ev
        state["events"] = keep

        existing = set(state["events"].keys())
        state["sent"] = {k:v for k,v in state["sent"].items() if k.split("|",1)[0] in existing or k.startswith("digest|")}

        cfg = state.get("cfg", DEFAULT_CFG)
        silent = bool(cfg.get("silent_default", True))

        # отправляем только двум (если они указаны)
        if USER1_CHAT_ID.isdigit() and USER2_CHAT_ID.isdigit():
            u1 = int(USER1_CHAT_ID)
            u2 = int(USER2_CHAT_ID)

            # ежедневный список "на сегодня"
            if due_daily_digest(cfg, n):
                key = f"digest|{n.date().isoformat()}"
                if not state["sent"].get(key):
                    today = n.date()
                    evs_today = []
                    for ev in state["events"].values():
                        dt = datetime.fromisoformat(ev["dt_iso"]).astimezone(TZ)
                        if dt.date() == today:
                            evs_today.append(ev)
                    evs_today.sort(key=lambda x: x["dt_iso"])
                    if evs_today:
                        txt = "События на сегодня:\n\n" + "\n".join(format_event(e) for e in evs_today)
                    else:
                        txt = "Событий на сегодня нет."
                    send(u1, txt, silent=silent)
                    send(u2, txt, silent=silent)
                    state["sent"][key] = n.isoformat()

            # напоминания по событиям
            for ev in state["events"].values():
                ev_dt = datetime.fromisoformat(ev["dt_iso"]).astimezone(TZ)
                for off in cfg.get("offsets_days", [1,0]):
                    sent_key = f"{ev['id']}|{int(off)}"
                    if state["sent"].get(sent_key):
                        continue
                    if due_for_offset(ev_dt, int(off), cfg, n):
                        dt_s = ev_dt.strftime("%d.%m.%Y %H:%M") if ev.get("has_time") else ev_dt.strftime("%d.%m.%Y")
                        label = f"Список {ev['slot']}: "
                        msg_text = (f"{label}Сегодня: {dt_s} — {ev['title']}"
                                    if int(off) == 0 else
                                    f"{label}Через {int(off)} дн.: {dt_s} — {ev['title']}")
                        send(u1, msg_text, silent=silent)
                        send(u2, msg_text, silent=silent)
                        state["sent"][sent_key] = n.isoformat()

        save_state(state_mid, state)

if __name__ == "__main__":
    main()

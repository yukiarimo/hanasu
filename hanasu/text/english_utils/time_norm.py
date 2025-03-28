import re
import inflect

_inflect = inflect.engine()
_time_re = re.compile(r"\b((0?[0-9])|(1[0-1])|(1[2-9])|(2[0-3])):([0-5][0-9])\s*(a\.m\.|am|pm|p\.m\.|a\.m|p\.m)?\b", re.IGNORECASE | re.X)

def expand_time_english(text: str) -> str:
    def _expand_time_english(match: "re.Match") -> str:
        hour = int(match.group(1))
        past_noon = hour >= 12

        if hour > 12:
            hour -= 12
        elif hour == 0:
            hour = 12
            past_noon = True

        time = [_inflect.number_to_words(hour)]

        minute = int(match.group(6))
        if minute > 0:
            if minute < 10:
                time.append("oh")
            time.append(_inflect.number_to_words(minute))

        am_pm = match.group(7)
        if am_pm is None:
            time.append("p m" if past_noon else "a m")
        else:
            time.extend(list(am_pm.replace(".", "")))

        return " ".join(time)

    return re.sub(_time_re, _expand_time_english, text)
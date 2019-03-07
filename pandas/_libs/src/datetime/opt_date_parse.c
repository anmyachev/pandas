#include "opt_date_parse.h"

#include <datetime.h>
#include <string.h>

static int inline parse_4digit(const char* s) {
     const char *ch = s;
     int result = 0;
     if (*ch < '0' || *ch > '9') return -1;
     result += (int)(*ch - '0') * 1000;
     ch++;
     if (*ch < '0' || *ch > '9') return -1;
     result += (int)(*ch - '0') * 100;
     ch++;
     if (*ch < '0' || *ch > '9') return -1;
     result += (int)(*ch - '0') * 10;
     ch++;
     if (*ch < '0' || *ch > '9') return -1;
     result += (int)(*ch - '0');
     return result;
}

static int inline parse_2digit(const char* s) {
    const char *ch = s;
    int result = 0;
    if (*ch < '0' || *ch > '9') return -1;
    result += (int)(*ch - '0') * 10;
    ch++;
    if (*ch < '0' || *ch > '9') return -1;
    result += (int)(*ch - '0');
    return result;
}

static int inline parse_1digit(const char* s) {
    if (*s < '0' || *s > '9') return -1;
    return (int)(*s - '0');
}


int index_Q(const char* string_with_index, int start_position,
            int end_position, int length) {
    const char* ch = NULL;
    int i;

    if (length > 0) {
        if (start_position > end_position) return -2;
        if (start_position < 0) return -3;
        if (end_position > length) end_position = length;

        ch = string_with_index + start_position;
        for (i = start_position; i < end_position; ++i, ++ch) {
            if ((*ch) == 'Q' ||  (*ch) == 'q') return i;
        }
    }
    return -4;
}

int parse_date_quarter(PyObject* parse_string, int* year, int* quarter) {
    const char* buf = NULL;
    int length, index;
    int _year, _quarter;
    int result;

    if (!PyUnicode_CheckExact(parse_string) ||
            !PyUnicode_IS_READY(parse_string)) {
        return -1;
    }
    buf = PyUnicode_DATA(parse_string);
    length = (int)PyUnicode_GET_LENGTH(parse_string);

    index = index_Q(buf, 1, 6, length);
    if (index == 1) {
        _quarter = parse_1digit(buf);
        if ((length == 4) || ((length == 5) && (buf[index + 1] == '-'))) {
            // r'(\d)Q-?(\d\d)')
            _year = 2000 + parse_2digit(buf + length - 2);
        } else if ((length == 6) || ((length == 7) &&
                                                (buf[index + 1] == '-'))) {
            // r'(\d)Q-?(\d\d\d\d)')
            _year = parse_4digit(buf + length - 4);
        } else {
            return -1;
        }
    } else if ((index == 2) || (index == 3)) {
        // r'(\d\d)-?Q(\d)'
        if ((length == 4) || ((length == 5) && (buf[index - 1] == '-'))) {
            _quarter = parse_1digit(buf + length - 1);
            _year = 2000 + parse_2digit(buf + length - 2);
        } else {
            return -1;
        }
    } else if ((index == 4) || (index == 5)) {
        // r'(\d\d\d\d)-?Q(\d)'
        if ((length == 6) || ((length == 7) && (buf[index - 1] == '-'))) {
            _quarter = parse_1digit(buf + length - 1);
            _year = parse_4digit(buf);
        } else {
            return -1;
        }
    } else {
        return -1;
    }

    result = ((_year != -1 || (_year > 2000)) && (_quarter != -1)) ? index : -1;
    if (result != -1) {
        *year = _year;
        *quarter = _quarter;
    }
    return result;
}

static char delimiters[4] = " /-\\";

int parse_month_year_date(PyObject* parse_string, int* year, int* month) {
    const char* buf = NULL;
    int length;
    int _year, _month;
    int result;

    if (!PyUnicode_CheckExact(parse_string) ||
            !PyUnicode_IS_READY(parse_string)) {
        return -1;
    }
    buf = PyUnicode_DATA(parse_string);
    length = (int)PyUnicode_GET_LENGTH(parse_string);

    if (length == 7) {
        const int delim1 = buf[2];
        const int delim2 = buf[4];
        if (strchr(delimiters, delim1) != NULL) {
            _month = parse_2digit(buf);
            _year = parse_4digit(buf + 3);
        } else if (strchr(delimiters, delim2) != NULL) {
            _year = parse_4digit(buf);
            _month = parse_2digit(buf + 5);
        } else {
            return -1;
        }
    } else {
        return -1;
    }
    result = ((_year != -1) && (_month != -1)) ? 0 : -1;
    if (result != -1) {
        *year = _year;
        *month = _month;
    }
    return result;
}

int parse_date_with_freq(PyObject* parse_string, PyObject* freq,
                        PyObject* compare_with_freq, int* year, int* month) {
    const char* buf = NULL;
    int length;
    int has_freq = 0;
    int _year, _month;
    int result;

    if (freq == Py_None) {
        return -1;
    }
    if (!PyUnicode_CheckExact(parse_string) ||
            !PyUnicode_IS_READY(parse_string)) {
        return -1;
    }
    buf = PyUnicode_DATA(parse_string);
    length = (int)PyUnicode_GET_LENGTH(parse_string);

    if (length == 6) {
        if (PyObject_RichCompareBool(freq, compare_with_freq, Py_EQ) == 1) {
            has_freq = 1;
        } else {
            PyObject* getattr_result = PyObject_GetAttrString(freq,
                                                              "rule_code");
            if (getattr_result == NULL) {
                PyErr_Clear();
                return -1;
            }
            has_freq = PyObject_RichCompareBool(getattr_result,
                                                compare_with_freq, Py_EQ) == 1;
            Py_DECREF(getattr_result);
        }
        if (has_freq) {
            _year = parse_4digit(buf);
            _month = parse_2digit(buf + 4);
            result = ((_year != -1) && (_month != -1)) ? 0 : -1;
            if (result != -1) {
                *year = _year;
                *month = _month;
            }
            return result;
        }
    }
    return -1;
}

PyObject* parse_slashed_date(PyObject* parse_string, PyObject* dayfirst,
                             PyObject* tzinfo, PyObject* DateParseError) {
    char* buf;
    Py_ssize_t length;
    int day, month, year;
    PyObject* result;
#if PY_MAJOR_VERSION == 2
#error Implement me
#else
    if (!PyUnicode_CheckExact(parse_string) ||
            !PyUnicode_IS_READY(parse_string)) {
        Py_RETURN_NONE;
    }
    buf = PyUnicode_DATA(parse_string);
    length = PyUnicode_GET_LENGTH(parse_string);
#endif
    if (length != 10 || strchr(delimiters, buf[2]) == NULL ||
                            strchr(delimiters, buf[5]) == NULL) {
        Py_RETURN_NONE;
    }

    {
        int part1, part2;
        if ((part1 = parse_2digit(buf)) == -1 ||
                (part2 = parse_2digit(&buf[3])) == -1 ||
                (year = parse_4digit(&buf[6])) == -1) {
            Py_RETURN_NONE;
        }
        switch (PyObject_IsTrue(dayfirst)) {
            case 1:
                day = part1;
                month = part2;
                break;
            case 0:
                month = part1;
                day = part2;
                break;
            default:
                return NULL;
        }
    }
    // smoke-validate day and month to throw away values that can never be valid
    if (day < 1 || day > 31 || month < 1 || month > 12) {
        return PyErr_Format(DateParseError,
                            "Invalid day (%d) or month (%d) specified",
                            day, month);
    }

    if (PyDateTimeAPI == NULL) {
        PyDateTime_IMPORT;
        if (PyDateTimeAPI == NULL) {
            return NULL;
        }
    }

    result = PyDateTimeAPI->DateTime_FromDateAndTime(year, month, day, 0, 0, 0,
            0, tzinfo, PyDateTimeAPI->DateTimeType);
    if (result == NULL) {
        return PyErr_Format(DateParseError,
                "Invalid day (%d), month (%d) or year (%d) specified", day,
                month, year);
    }
    return result;
}

int does_string_look_like_time(PyObject* parse_string) {
    char* buf;
    Py_ssize_t length;
    int hour, minute;
#if PY_MAJOR_VERSION == 2
#error Implement me
#else
    if (!PyUnicode_CheckExact(parse_string) ||
            !PyUnicode_IS_READY(parse_string)) {
        // not a string, so doesn't look like time
        return 0;
    }
    buf = PyUnicode_DATA(parse_string);
    length = PyUnicode_GET_LENGTH(parse_string);
#endif
    if (length < 4) {
        // h:MM doesn't fit in, not a time
        return 0;
    }
    if (buf[1] == ':') {
        // h:MM format
        hour = parse_1digit(buf);
        minute = parse_2digit(&buf[2]);
    } else if (buf[2] == ':') {
        // HH:MM format
        hour = parse_2digit(buf);
        minute = parse_2digit(&buf[3]);
    } else {
        // not a time
        return 0;
    }

    return (hour >= 0 && hour <= 23 && minute >= 0 && minute <= 59) ? 1 : 0;
}

PyObject* make_date_from_year_month(int year, int month, PyObject* default_date,
                                    PyObject* default_tzinfo) {
    if (default_date == Py_None) {
        if (PyDateTimeAPI == NULL) {
            PyDateTime_IMPORT;
            if (PyDateTimeAPI == NULL) {
                return NULL;
            }
        }
        return PyDateTimeAPI->DateTime_FromDateAndTime(year, month, 1, 0, 0, 0,
                0, default_tzinfo, PyDateTimeAPI->DateTimeType);
    } else {
        PyObject* replace_meth = PyObject_GetAttrString(default_date,
                                                        "replace");
        PyObject* result = NULL;
        PyObject* kw;
        PyObject* pyYear;
        PyObject* pyMonth;

        if (replace_meth == NULL) return NULL;
        kw = PyDict_New();
        if (kw == NULL) {
            Py_DECREF(replace_meth);
            return NULL;
        }
        pyYear = PyLong_FromLong(year);
        if (pyYear == NULL) {
            Py_DECREF(replace_meth);
            Py_DECREF(kw);
            return NULL;
        }
        pyMonth = PyLong_FromLong(month);
        if (pyMonth == NULL) {
            Py_DECREF(replace_meth);
            Py_DECREF(kw);
            Py_DECREF(pyYear);
            return NULL;
        }
        if ((PyDict_SetItemString(kw, "month", pyMonth) == 0) &&
                (PyDict_SetItemString(kw, "year", pyYear) == 0)) {
            PyObject* emptyTuple = PyTuple_New(0);
            if (emptyTuple != NULL) {
                result = PyObject_Call(replace_meth, emptyTuple, kw);
                Py_DECREF(emptyTuple);
            }
        }
        Py_DECREF(replace_meth);
        Py_DECREF(kw);
        Py_DECREF(pyYear);
        Py_DECREF(pyMonth);
        return result;
    }
}

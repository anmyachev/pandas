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


int index_Q(const char* string, int start_position, int end_position, int length) {
    const char* ch = NULL;
    int i;

    if (length > 0) {
        if (start_position > end_position) {
            return -2;
        }
        if ((start_position < 0) || (end_position > length)) {
            return -3;
        }
        ch = string + start_position;
        for (i = start_position; i < end_position; ++i, ++ch) {
            if ((*ch) == 'Q' ||  (*ch) == 'q') return i;
        }
    }
    return -4;
}

int parse_date_quarter(PyObject* string, int* year, int* quarter) {
    const char* buf = NULL;
    int length, index;

    if (!PyUnicode_CheckExact(string) || !PyUnicode_IS_READY(string)) {
        return -1;
    }
    buf = PyUnicode_DATA(string);
    length = (int)PyUnicode_GET_LENGTH(string);

    index = index_Q(buf, 1, 6, length);
    if (index == 1) {
        *quarter = parse_1digit(buf);
        if ((length == 4) || ((length == 5) && (buf[index + 1] == '-'))) {
            // r'(\d)Q-?(\d\d)')
            *year = 2000 + parse_2digit(buf + length - 2);
        } else if ((length == 6) || ((length == 7) && (buf[index + 1] == '-'))) {
            // r'(\d)Q-?(\d\d\d\d)')
            *year = parse_4digit(buf + length - 4);
        } else {
            return -1;
        }
    } else if ((index == 2) || (index == 3)){
        // r'(\d\d)-?Q(\d)'
        if ((length == 4) || ((length == 5) && (buf[index - 1] == '-'))) {
            *quarter = parse_1digit(buf + length - 1);
            *year = 2000 + parse_2digit(buf + length - 2);
        } else {
            return -1;
        }
    } else if ((index == 4) || (index == 5)) {
        // r'(\d\d\d\d)-?Q(\d)'
        if ((length == 6) || ((length == 7) && (buf[index - 1] == '-'))) {
            *quarter = parse_1digit(buf + length - 1);
            *year = parse_4digit(buf);
        } else {
            return -1;
        }
    } else return -1;
    return (((*year) != -1 || ((*year) > 2000)) && ((*quarter) != -1)) ? index : -1;
}

static char delimiters[4] = " /-\\";

int parse_month_year_date(PyObject* string, int* year, int* month)
{
    const char* buf = NULL;
    int length;

    if (!PyUnicode_CheckExact(string) || !PyUnicode_IS_READY(string)) {
        return -1;
    }
    buf = PyUnicode_DATA(string);
    length = (int)PyUnicode_GET_LENGTH(string);

    if (length == 7) {
        const int delim1 = buf[2];
        const int delim2 = buf[4];
        if (strchr(delimiters, delim1) != NULL) {
            *month = parse_2digit(buf);
            *year = parse_4digit(buf + 3);
        } else if (strchr(delimiters, delim2) != NULL) {
            *year = parse_4digit(buf);
            *month = parse_2digit(buf + 5);
        } else {
            return -1;
        }
    } else {
        return -1;
    }

    return (((*year) != -1) && ((*month) != -1)) ? 0 : -1;
}

int parse_date_with_freq(PyObject* string, PyObject* freq, PyObject* compare_with_freq, int* year, int* month) {
    const char* buf = NULL;
    PyObject* getattr_result = NULL;
    int length;

    if (!PyUnicode_CheckExact(string) || !PyUnicode_IS_READY(string)) {
        return -1;
    }
    buf = PyUnicode_DATA(string);
    length = (int)PyUnicode_GET_LENGTH(string);

    if ((length == 6) && ((PyObject_RichCompareBool(freq, compare_with_freq, Py_EQ) == 1) ||
            (PyObject_RichCompareBool((getattr_result = PyObject_GetAttrString(freq, "rule_code")), compare_with_freq, Py_EQ) == 1))) {
        *year = parse_4digit(buf);
        *month = parse_2digit(buf + 5);
    }
    Py_XDECREF(getattr_result);
    return (((*year) != -1) && ((*month) != -1)) ? 0 : -1;
}

int does_string_look_like_time(PyObject* string)
{
    char* buf;
    Py_ssize_t length;
    int hour, minute;
#if PY_MAJOR_VERSION == 2
#error Implement me
#else
    if (!PyUnicode_CheckExact(string) || !PyUnicode_IS_READY(string)) {
        // not a string, so doesn't look like time
        return 0;
    }
    buf = PyUnicode_DATA(string);
    length = PyUnicode_GET_LENGTH(string);
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

PyObject* make_date_from_year_month(int year, int month, PyObject* default_date, PyObject* default_tzinfo) {
    if (default_date == Py_None) {
        if (PyDateTimeAPI == NULL) {
            PyDateTime_IMPORT;
            if (PyDateTimeAPI == NULL) {
                return NULL;
            }
        }
        return PyDateTimeAPI->DateTime_FromDateAndTime(year, month, 1, 0, 0, 0, 0, default_tzinfo, PyDateTimeAPI->DateTimeType);
    } else {
        PyObject* replace_meth = PyObject_GetAttrString(default_date, "replace");
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
        if ((PyDict_SetItemString(kw, "month", pyMonth) == 0) && (PyDict_SetItemString(kw, "year", pyYear) == 0)) {
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

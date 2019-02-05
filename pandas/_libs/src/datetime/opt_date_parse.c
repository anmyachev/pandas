#include "opt_date_parse.h"

Py_ssize_t index_Q(const char* string, Py_ssize_t start_position, Py_ssize_t end_position, Py_ssize_t length) {
    char* buf = NULL;
    char* ch = NULL;

    if (length > 0) {
        if (start_position > end_position) {
            return -2;
        }
        if ((start_position < 0 || start_position > length) || (end_position > length)) {
            return -3;
        }
        ch = buf + start_position;
        for (Py_ssize_t i = start_position; i < end_position; ++i, ++ch) {
            if ((*ch) == 'Q' ||  (*ch) == 'q') return i;
        }
    }
    return -4;
}

int parse_date_quarter(PyObject* string, int* year, int* quarter) {
    char* buf = NULL;
    char* ch = NULL;
    Py_ssize_t length = -1;
    Py_ssize_t index;

    if (!PyUnicode_CheckExact(string) || !PyUnicode_IS_READY(string)) {
        return -1;
    }
    buf = PyUnicode_DATA(string);
    length = PyUnicode_GET_LENGTH(string);

    index = index_Q(buf, 1, 6, length);
    if (index == 1) {
        *quarter = buf[0] - '0';
        //if ((length == 4) || (length == 5))
    }
    return 0;
}





/*i = date_string.index('Q', 1, 6)
            if i == 1:
                quarter = int(date_string[0])
                if date_len == 4 or (date_len == 5
                                     and date_string[i + 1] == '-'):
                    # r'(\d)Q-?(\d\d)')
                    year = 2000 + int(date_string[-2:])
                elif date_len == 6 or (date_len == 7
                                       and date_string[i + 1] == '-'):
                    # r'(\d)Q-?(\d\d\d\d)')
                    year = int(date_string[-4:])
                else:
                    raise ValueError
            elif i == 2 or i == 3:
                # r'(\d\d)-?Q(\d)'
                if date_len == 4 or (date_len == 5
                                     and date_string[i - 1] == '-'):
                    quarter = int(date_string[-1])
                    year = 2000 + int(date_string[:2])
                else:
                    raise ValueError
            elif i == 4 or i == 5:
                if date_len == 6 or (date_len == 7
                                     and date_string[i - 1] == '-'):
                    # r'(\d\d\d\d)-?Q(\d)'
                    quarter = int(date_string[-1])
                    year = int(date_string[:4])
                else:
                    raise ValueError*/
#include <Python.h>

int index_Q(const char* string, int start_position, int end_position, int length);

int parse_month_year_date(PyObject* string, int* year, int* month);
int parse_date_with_freq(PyObject* string, PyObject* freq, PyObject* compare_with_freq, int* year, int* month);
int parse_date_quarter(PyObject* string, int* year, int* quarter);
int does_string_look_like_time(PyObject* string);

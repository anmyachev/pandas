#include <Python.h>

Py_ssize_t index_Q(const char* string, Py_ssize_t start_position, Py_ssize_t end_position, Py_ssize_t length);

int parse_date_quarter(PyObject* string, int* year, int* quarter);
int does_string_look_like_time(PyObject* string);

#ifndef PANDAS__LIBS_TSLIBS_SRC_DATETIME_OPT_DATE_PARSE_H_
#define PANDAS__LIBS_TSLIBS_SRC_DATETIME_OPT_DATE_PARSE_H_

#include <Python.h>

int parse_month_year_date(PyObject* parse_string, int* year, int* month);
int parse_date_quarter(PyObject* parse_string, int* year, int* quarter);
int parse_date_with_freq(PyObject* parse_string, PyObject* freq,
                        PyObject* compare_with_freq, int* year, int* month);

PyObject* parse_slashed_date(PyObject* parse_string, PyObject* dayfirst,
                            PyObject* tzinfo, PyObject* DateParseError);

int does_string_look_like_time(PyObject* parse_string);

PyObject* make_date_from_year_month(int year, int month, PyObject* default_date,
                                    PyObject* default_tzinfo);

#endif  // PANDAS__LIBS_TSLIBS_SRC_DATETIME_OPT_DATE_PARSE_H_

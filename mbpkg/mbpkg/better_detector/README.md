# Better Detector

## Why

Better Detector is an improved version of Detector.
It is separate from 'old' detector because it uses a separate representation
for data, which is not compatible with old detector's ones (as well as
files created using it).

Better detector is still called 'Detector' because the next goal
is to replace old detector with this one.
At that point, this package will be renamed.


## API
Detector does the following:
- create a detector from different objects using `Detector(x)`:
    - a `SPOT_DETECTION_METHOD` Keyword str
    - a known `Callable`
    - a deserializable `str` (starts with '@' character)
- convert it to a factory-compatible representation: `Detector.to_factory()`
- try to convert it to a `functools.partial`: `Detector.try_to_partial()`
- convert it to a textual representation: `Detector.to_str()`
- try to convert it to a kw `str`: `Detector.try_to_kwstr()`


(a `Detector.try_to_callable()` might be implemented later, but for now all
deserialized-supported callables are partials)

Note: 
- contrary to legacy Detector, this one does not accept functions
that cannot be deserialized automatically.
- dict init has been removed for now


## str textual representation
str textual representation can be one of the following:
- a `SPOT_DETECTION_METHOD` Keyword str
- a known `Callable` `str` starting with '@'

In the second case the representation is the following:
"@{func_name}|{arg1}={v1}|{arg2}={v2}|{arg3}={v3}"

Values are encoded that way:
- True: T
- False: F
- int: i{int_value}
- float: f{float_value}

> for example this is lapgau's representation:
`@mm_log|min_sigma=i5|max_sigma=i10|num_sigma=i5|threshold=f0.01`



# Detector package

The dector package provides a representation for the MidBodyFactory detectors.
It can be used to:
- load detectors from their str representation
- dynamically create detectors
- save these detectors to disk


The main element of this package is the `Detector` class. With this, you can:
- create a `Detector` from its string representataion: `Detector.from_str`
- convert a `Detector` to its string representation: `Detector.to_str`
- convert a `Detector` to a factory callable: `Detector.to_factory`

You can also wrap a factory spot detection method in a `Detector` with
`Detector.from_factory`
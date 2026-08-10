from fieldmate.brain.pronounce import tts_pronounce


def test_temperature_conversion_pronunciation():
    assert tts_pronounce("10°C is 50°F") == "10 degrees Celsius is 50 degrees Fahrenheit"
    assert tts_pronounce("Convert 10 °C") == "Convert 10 degrees Celsius"


def test_technical_acronyms_pronunciation():
    assert tts_pronounce("Check the BSOD log and UEFI settings") == "Check the B. S. O. D. log and U. E. F. I. settings"
    assert tts_pronounce("NVMe drive speed is 7000 MB/s") == "N. V. M. E. drive speed is 7000 megabytes per second"


def test_symbol_normalizations():
    assert tts_pronounce("CPU usage is at 95%") == "C. P. U. usage is at 95 percent"
    assert tts_pronounce("Resolution is 1920x1080") == "Resolution is 1920 by 1080"
    assert tts_pronounce("Latency ~ 50ms") == "Latency approximately 50 milliseconds"
    assert tts_pronounce("Value = 10 — done") == "Value equals 10, done"

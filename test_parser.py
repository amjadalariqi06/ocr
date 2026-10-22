import unittest
from main import parse_document, normalize_digits

class ParserTests(unittest.TestCase):
    def test_arabic_digits(self):
        self.assertEqual(normalize_digits("١٢٣۴۵۶"), "123456")

    def test_labeled_visa_fields(self):
        result = parse_document("تأشيرة العمرة\nName: AMJAD AHMED\nPassport No: AB1234567\nVisa No: 6123456789\nDate of Issue: 12/10/2026")
        self.assertEqual(result["full_name"], "AMJAD AHMED")
        self.assertEqual(result["passport_number"], "AB1234567")
        self.assertEqual(result["reference_number"], "6123456789")
        self.assertEqual(result["date"], "12/10/2026")

    def test_does_not_guess_arbitrary_long_number(self):
        result = parse_document("مستند غير معروف\n123456789012\nلا يوجد رقم جواز أو تذكرة واضح")
        self.assertEqual(result["reference_number"], "غير متاح")
        self.assertEqual(result["passport_number"], "غير متاح")

    def test_air_ticket(self):
        result = parse_document("Yemenia\nPASSENGER NAME: AL-ARIQI/AMJAD\nTICKET NUMBER: 6351234567890\nDATE: 23.OCT.2026")
        self.assertEqual(result["document_type"], "تذكرة طيران")
        self.assertIn("AL-ARIQI", result["full_name"])
        self.assertEqual(result["reference_number"], "6351234567890")

if __name__ == '__main__':
    unittest.main()

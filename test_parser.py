import unittest
from main import parse_universal_document

class ParserRegressionTests(unittest.TestCase):
    def test_visa_fields(self):
        result = parse_universal_document(
            "المملكة العربية السعودية\nName: AHMED ALI\n"
            "Passport No: A1234567\nVisa No: 61234567890\nIssue Date: 12/10/2026"
        )
        self.assertEqual(result["full_name"], "AHMED ALI")
        self.assertEqual(result["passport_number"], "A1234567")
        self.assertEqual(result["reference_number"], "61234567890")
        self.assertEqual(result["date"], "12/10/2026")

    def test_yemenia_ticket(self):
        result = parse_universal_document(
            "Yemenia Airways\nPASSENGER NAME: AHMED ALI\n"
            "TICKET NUMBER: 6351234567890\n12.OCT.2026"
        )
        self.assertEqual(result["full_name"], "AHMED ALI")
        self.assertEqual(result["reference_number"], "6351234567890")
        self.assertEqual(result["document_type"], "تذكرة طيران دولية (تصنيف آلي)")

    def test_arabic_digits(self):
        result = parse_universal_document("رقم التأشيرة: ٦١٢٣٤٥٦٧٨٩٠")
        self.assertEqual(result["reference_number"], "61234567890")

if __name__ == "__main__":
    unittest.main()

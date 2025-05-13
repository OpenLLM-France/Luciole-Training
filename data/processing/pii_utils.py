
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter

class PhoneNumberPII(BaseFilter):
    name = "📞 Phone Number PII"
    """"
    This filter uses the phonenumbers library to find and replace phone numbers in the text.
    It also stores the original phone numbers in the metadata of the document.
    The replacement text is <<pii_phone_number>> by default.
    The country code is set to US by default, but can be changed by passing a different country code.

    Example of country codes: "US", "GB", "FR", "DE", "IT", "ES", "PT", "NL"
    """

    def __init__(
        self,
        country: str = "US",
        replacement: str = "<<pii_phone_number>>",
        exclusion_writer: DiskWriter = None,
    ):
        super().__init__(exclusion_writer)
        self.country = country
        self.replacement = replacement
        import phonenumbers

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        if 'pii_phone_number' not in doc.metadata:
            doc.metadata['pii_phone_number'] = []
        matches = list(phonenumbers.PhoneNumberMatcher(doc.text, country))
        # Replace from the end to the beginning to avoid messing up indices
        new_text = doc.text
        for m in reversed(matches):
            doc.metadata['pii_phone_number'].append(str(m))
            new_text = new_text[:m.start] + self.replacement + new_text[m.end:]
        doc.text = new_text
        return True

# Too slow but can be used to label small subset of data
class PresidioPII(BaseFilter):
    name = "🏛️ Presidio PII"

    def __init__(
        self,
        exclusion_writer: DiskWriter = None,
    ):
        super().__init__(exclusion_writer)
        self._analyzer = None

    @property
    def analyzer(self):
        if self._analyzer is None:
            from presidio_analyzer import AnalyzerEngine
            self._analyzer = AnalyzerEngine()
        return self._analyzer

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        analyzer_results = self.analyzer.analyze(text=doc.text, language='en') # , entities=["PHONE_NUMBER"]
        doc.metadata['presidio'] = [x.to_dict() for x in analyzer_results]
        return True
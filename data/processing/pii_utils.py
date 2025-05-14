
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
import re
from functools import partial
from typing import Callable
from datatrove.pipeline.formatters.base import BaseFormatter
from datatrove.pipeline.formatters.pii import PIIReplacer

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
        replacement: str = "<<pii_phone>>",
        exclusion_writer: DiskWriter = None,
    ):
        super().__init__(exclusion_writer)
        self.country = country
        self.replacement = replacement

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        import phonenumbers
        if 'pii_phone_number' not in doc.metadata:
            doc.metadata['pii_phone_number'] = []
        matches = list(phonenumbers.PhoneNumberMatcher(doc.text, self.country))
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

class MorePIIFormatter(BaseFormatter):
    """
    Replaces NIR in the document text.
    Args:
    """

    name = "💳 More PII"

    def __init__(
        self,
        remove_nir: bool = True,
        remove_url: bool = False,
        nir_replacement: str = "<<pii_nir>>",
        url_replacement: str = "<<url_link>>",
    ):
        super().__init__()
        self.remove_nir = remove_nir
        self.remove_url = remove_url

        BASE_URL_REGEX = r"((www\d{0,3}[.])?[a-z0-9.\-]+[.](?:(?:com)|(?:edu)|(?:gov)|(?:int)|(?:mil)|(?:net)|(?:onl)|(?:org)|(?:pro)|(?:red)|(?:tel)|(?:uno)|(?:xxx)|(?:academy)|(?:accountant)|(?:accountants)|(?:actor)|(?:adult)|(?:africa)|(?:agency)|(?:airforce)|(?:apartments)|(?:app)|(?:archi)|(?:army)|(?:art)|(?:asia)|(?:associates)|(?:attorney)|(?:auction)|(?:audio)|(?:auto)|(?:autos)|(?:baby)|(?:band)|(?:bar)|(?:bargains)|(?:beer)|(?:berlin)|(?:best)|(?:bet)|(?:bid)|(?:bike)|(?:bio)|(?:black)|(?:blackfriday)|(?:blog)|(?:blue)|(?:boats)|(?:bond)|(?:boo)|(?:boston)|(?:bot)|(?:boutique)|(?:build)|(?:builders)|(?:business)|(?:buzz)|(?:cab)|(?:cafe)|(?:cam)|(?:camera)|(?:camp)|(?:capital)|(?:car)|(?:cards)|(?:care)|(?:careers)|(?:cars)|(?:casa)|(?:cash)|(?:casino)|(?:catering)|(?:center)|(?:ceo)|(?:cfd)|(?:charity)|(?:chat)|(?:cheap)|(?:christmas)|(?:church)|(?:city)|(?:claims)|(?:cleaning)|(?:click)|(?:clinic)|(?:clothing)|(?:cloud)|(?:club)|(?:codes)|(?:coffee)|(?:college)|(?:com)|(?:community)|(?:company)|(?:computer)|(?:condos)|(?:construction)|(?:consulting)|(?:contact)|(?:contractors)|(?:cooking)|(?:cool)|(?:coupons)|(?:courses)|(?:credit)|(?:creditcard)|(?:cricket)|(?:cruises)|(?:cyou)|(?:dad)|(?:dance)|(?:date)|(?:dating)|(?:day)|(?:degree)|(?:delivery)|(?:democrat)|(?:dental)|(?:dentist)|(?:desi)|(?:design)|(?:dev)|(?:diamonds)|(?:diet)|(?:digital)|(?:direct)|(?:directory)|(?:discount)|(?:doctor)|(?:dog)|(?:domains)|(?:download)|(?:earth)|(?:eco)|(?:education)|(?:email)|(?:energy)|(?:engineer)|(?:engineering)|(?:enterprises)|(?:equipment)|(?:esq)|(?:estate)|(?:events)|(?:exchange)|(?:expert)|(?:exposed)|(?:express)|(?:fail)|(?:faith)|(?:family)|(?:fans)|(?:farm)|(?:fashion)|(?:feedback)|(?:film)|(?:finance)|(?:financial)|(?:fish)|(?:fishing)|(?:fit)|(?:fitness)|(?:flights)|(?:florist)|(?:flowers)|(?:football)|(?:forsale)|(?:foundation)|(?:fun)|(?:fund)|(?:furniture)|(?:futbol)|(?:fyi)|(?:gallery)|(?:game)|(?:games)|(?:garden)|(?:gay)|(?:gdn)|(?:gifts)|(?:gives)|(?:giving)|(?:glass)|(?:global)|(?:gmbh)|(?:gold)|(?:golf)|(?:graphics)|(?:gratis)|(?:green)|(?:gripe)|(?:group)|(?:guide)|(?:guitars)|(?:guru)|(?:hair)|(?:hamburg)|(?:haus)|(?:health)|(?:healthcare)|(?:help)|(?:hiphop)|(?:hockey)|(?:holdings)|(?:holiday)|(?:homes)|(?:horse)|(?:hospital)|(?:host)|(?:hosting)|(?:house)|(?:how)|(?:icu)|(?:info)|(?:ink)|(?:institute)|(?:insure)|(?:international)|(?:investments)|(?:irish)|(?:jewelry)|(?:jetzt)|(?:juegos)|(?:kaufen)|(?:kids)|(?:kitchen)|(?:kiwi)|(?:krd)|(?:kyoto)|(?:land)|(?:lat)|(?:law)|(?:lawyer)|(?:lease)|(?:legal)|(?:lgbt)|(?:life)|(?:lighting)|(?:limited)|(?:limo)|(?:link)|(?:live)|(?:loan)|(?:loans)|(?:lol)|(?:london)|(?:love)|(?:ltd)|(?:ltda)|(?:luxury)|(?:maison)|(?:management)|(?:market)|(?:marketing)|(?:markets)|(?:mba)|(?:media)|(?:melbourne)|(?:meme)|(?:memorial)|(?:men)|(?:miami)|(?:mobi)|(?:moda)|(?:moe)|(?:mom)|(?:money)|(?:monster)|(?:mortgage)|(?:motorcycles)|(?:mov)|(?:movie)|(?:nagoya)|(?:name)|(?:navy)|(?:network)|(?:new)|(?:news)|(?:ngo)|(?:ninja)|(?:now)|(?:nyc)|(?:observer)|(?:okinawa)|(?:one)|(?:ong)|(?:onl)|(?:online)|(?:organic)|(?:osaka)|(?:page)|(?:paris)|(?:partners)|(?:parts)|(?:party)|(?:pet)|(?:phd)|(?:photo)|(?:photography)|(?:photos)|(?:pics)|(?:pictures)|(?:pink)|(?:pizza)|(?:place)|(?:plumbing)|(?:plus)|(?:poker)|(?:porn)|(?:press)|(?:pro)|(?:productions)|(?:prof)|(?:promo)|(?:properties)|(?:property)|(?:protection)|(?:pub)|(?:quest)|(?:racing)|(?:recipes)|(?:red)|(?:rehab)|(?:reise)|(?:reisen)|(?:rent)|(?:rentals)|(?:repair)|(?:report)|(?:republican)|(?:rest)|(?:restaurant)|(?:review)|(?:reviews)|(?:rip)|(?:rocks)|(?:rodeo)|(?:rsvp)|(?:run)|(?:saarland)|(?:sale)|(?:salon)|(?:sarl)|(?:sbs)|(?:school)|(?:schule)|(?:science)|(?:services)|(?:sex)|(?:sexy)|(?:sh)|(?:shoes)|(?:shop)|(?:shopping)|(?:show)|(?:singles)|(?:site)|(?:skin)|(?:soccer)|(?:social)|(?:software)|(?:solar)|(?:solutions)|(?:soy)|(?:space)|(?:spiegel)|(?:study)|(?:style)|(?:sucks)|(?:supply)|(?:support)|(?:surf)|(?:surgery)|(?:systems)|(?:tax)|(?:taxi)|(?:team)|(?:tech)|(?:technology)|(?:tel)|(?:theater)|(?:tips)|(?:tires)|(?:today)|(?:tools)|(?:top)|(?:tours)|(?:town)|(?:toys)|(?:trade)|(?:training)|(?:tube)|(?:uk)|(?:university)|(?:uno)|(?:vacations)|(?:ventures)|(?:vet)|(?:video)|(?:villas)|(?:vin)|(?:vip)|(?:vision)|(?:vlaanderen)|(?:vodka)|(?:vote)|(?:voting)|(?:voyage)|(?:wales)|(?:wang)|(?:watch)|(?:webcam)|(?:website)|(?:wedding)|(?:wiki)|(?:wine)|(?:work)|(?:works)|(?:world)|(?:wtf)|(?:xyz)|(?:yoga)|(?:yokohama)|(?:you)|(?:zone)|(?:ac)|(?:ad)|(?:ae)|(?:af)|(?:ag)|(?:ai)|(?:al)|(?:am)|(?:an)|(?:ao)|(?:aq)|(?:ar)|(?:as)|(?:at)|(?:au)|(?:aw)|(?:ax)|(?:az)|(?:ba)|(?:bb)|(?:bd)|(?:be)|(?:bf)|(?:bg)|(?:bh)|(?:bi)|(?:bj)|(?:bm)|(?:bn)|(?:bo)|(?:br)|(?:bs)|(?:bt)|(?:bv)|(?:bw)|(?:by)|(?:bz)|(?:ca)|(?:cc)|(?:cd)|(?:cf)|(?:cg)|(?:ch)|(?:ci)|(?:ck)|(?:cl)|(?:cm)|(?:cn)|(?:co)|(?:cr)|(?:cu)|(?:cv)|(?:cw)|(?:cx)|(?:cy)|(?:cz)|(?:de)|(?:dj)|(?:dk)|(?:dm)|(?:do)|(?:dz)|(?:ec)|(?:ee)|(?:eg)|(?:er)|(?:es)|(?:et)|(?:eu)|(?:fi)|(?:fj)|(?:fk)|(?:fm)|(?:fo)|(?:fr)|(?:ga)|(?:gb)|(?:gd)|(?:ge)|(?:gf)|(?:gg)|(?:gh)|(?:gi)|(?:gl)|(?:gm)|(?:gn)|(?:gp)|(?:gq)|(?:gr)|(?:gs)|(?:gt)|(?:gu)|(?:gw)|(?:gy)|(?:hk)|(?:hm)|(?:hn)|(?:hr)|(?:ht)|(?:hu)|(?:id)|(?:ie)|(?:il)|(?:im)|(?:in)|(?:io)|(?:iq)|(?:ir)|(?:is)|(?:it)|(?:je)|(?:jm)|(?:jo)|(?:jp)|(?:ke)|(?:kg)|(?:kh)|(?:ki)|(?:km)|(?:kn)|(?:kp)|(?:kr)|(?:kw)|(?:ky)|(?:kz)|(?:la)|(?:lb)|(?:lc)|(?:li)|(?:lk)|(?:lr)|(?:ls)|(?:lt)|(?:lu)|(?:lv)|(?:ly)|(?:ma)|(?:mc)|(?:md)|(?:me)|(?:mg)|(?:mh)|(?:mk)|(?:ml)|(?:mm)|(?:mn)|(?:mo)|(?:mp)|(?:mq)|(?:mr)|(?:ms)|(?:mt)|(?:mu)|(?:mv)|(?:mw)|(?:mx)|(?:my)|(?:mz)|(?:na)|(?:nc)|(?:ne)|(?:nf)|(?:ng)|(?:ni)|(?:nl)|(?:no)|(?:np)|(?:nr)|(?:nu)|(?:nz)|(?:om)|(?:pa)|(?:pe)|(?:pf)|(?:pg)|(?:ph)|(?:pk)|(?:pl)|(?:pm)|(?:pn)|(?:pr)|(?:ps)|(?:pt)|(?:pw)|(?:py)|(?:qa)|(?:re)|(?:ro)|(?:rs)|(?:ru)|(?:rw)|(?:sa)|(?:sb)|(?:sc)|(?:sd)|(?:se)|(?:sg)|(?:sh)|(?:si)|(?:sj)|(?:sk)|(?:sl)|(?:sm)|(?:sn)|(?:so)|(?:sr)|(?:st)|(?:su)|(?:sv)|(?:sx)|(?:sy)|(?:sz)|(?:tc)|(?:td)|(?:tf)|(?:tg)|(?:th)|(?:tj)|(?:tk)|(?:tl)|(?:tm)|(?:tn)|(?:to)|(?:tp)|(?:tr)|(?:tt)|(?:tv)|(?:tw)|(?:tz)|(?:ua)|(?:ug)|(?:uk)|(?:us)|(?:uy)|(?:uz)|(?:va)|(?:vc)|(?:ve)|(?:vg)|(?:vi)|(?:vn)|(?:vu)|(?:wf)|(?:ws)|(?:ye)|(?:yt)|(?:za)|(?:zm)|(?:zw))(?:/[^\s()<>\"']*)?)"  # noqa: E501
        combined_regex = (
            r'(?i)(?:https?://' + BASE_URL_REGEX + r'|'
            + BASE_URL_REGEX + r'|'
            r'["\']https?://' + BASE_URL_REGEX + r'["\']|'
            r'["\']' + BASE_URL_REGEX + r'["\'])'
        )
        self.url_replacer = PIIReplacer(
            combined_regex,
            replacements="<<url_link>>",
        )
        self.nir_replacer = PIIReplacer(
            r"[12]\s?\d{2}\s?(0[1-9]|1[0-2]|[2-9][0-9])\s?([0-9]{2}|2A|2B)\s?\d{3}\s?\d{3}\s?\d{2}",
            replacements=nir_replacement,
        )

    def format(self, text: str) -> str:
        if self.remove_nir:
            text = self.nir_replacer.replace(text)
        if self.remove_url:
            text = self.url_replacer.replace(text)
        return text
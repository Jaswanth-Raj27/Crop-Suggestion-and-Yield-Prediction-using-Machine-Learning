# Agriculture data

## FAOSTAT India crop-yield extract

`processed/india_faostat_crop_yield.csv` is a local, filtered extract of FAOSTAT's Crops and Livestock Products (QCL) bulk dataset. It contains India rows where `Element` is `Yield` and `Unit` is `kg/ha`. The extract keeps the source `Value`, `Flag`, and `Note` fields; use the flag when interpreting observations rather than treating every row as a directly measured official value.

The downloaded source archive is kept under the ignored `data/raw/` directory and is not intended for Git. It was downloaded from the FAO bulk endpoint:

<https://bulks-faostat.fao.org/production/Production_Crops_Livestock_E_All_Data_(Normalized).zip>

The archive's HTTP `Last-Modified` date was 31 December 2025. Its included metadata describes the QCL data as covering 1961–2024. The source archive SHA-256 at download was `C5835418C18F9322E7DECBD6800F93A216EAAE3CDFA31ACB08F0518C0C6D6853`.

### Attribution and license

FAO. 2025. *FAOSTAT: Crops and livestock products (QCL).* Accessed 26 October 2026. <https://www.fao.org/faostat/en/#data/QCL>. Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0): <https://creativecommons.org/licenses/by/4.0/>. FAO's statistical terms require attribution and caution that some source observations may come from third parties or be estimated/imputed; check the dataset metadata and flags when reusing the data. This project does not imply FAO endorsement.

### Important limitations

- FAOSTAT is annual, country-level data. This extract does not include Indian state/district, season, soil nutrients, pH, weather, or pest features.
- Yield units are retained as `kg/ha`. FAOSTAT defines yield as harvested production per hectare; crop outputs are not a like-for-like quality or profitability ranking. Fresh fruit/vegetable mass can exceed dry-grain mass per hectare.
- FAOSTAT values can be official, estimated, imputed, missing, or sourced externally. The extract preserves the source status flag for this reason.
- This extract can support a national historical-yield view or a carefully evaluated country-level baseline. It cannot train the current location-and-soil-based crop recommender as-is.
- The original `indian_agriculture_data.csv` is unchanged and remains synthetic demonstration data.

## Interface photography

Each exact FAOSTAT item label in processed/india_faostat_crop_yield.csv maps to its own distinct crop photo in data/crop_visuals.json. The mapping and tests ensure that no item shares an image URL or source page; local images are also checked for identical file hashes. Aggregate labels use a representative photo from that group. These images provide visual context only and are not the source of the yield values.

There are 96 mapped photos: 61 are served from this project's static files and 35 are loaded from Wikimedia Commons thumbnails. The remotely hosted images require an internet connection and may be unavailable if Commons is down or rate-limits the request. Credits, source pages, and license details for every item are recorded in static/images/crops/credits.json; keep the on-page photo attribution visible.
The scrolling library is [Lenis](https://github.com/darkroomengineering/lenis), version 1.3.27, served locally under `static/vendor/lenis/` and distributed under its MIT license.

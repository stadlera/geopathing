# geopathing
Geocoded pathing used to caculate distance/time-of-arrival maps using the google maps api.

This project is used to generate optimal distributions of operational areas for different fire brigades.
The base data is read in as a .csv and should contain a list of adresses (postcode, street, housenumber and latitude/longitude).
This data has to be generated in advance using the OpenStreetMap api and filtering all points-of-interest using the Heidelberg location code.

Afterwards requests targeted at the Google Maps Distance and Geocoding api are sent and used to generate a distance/time-of-arrival maps are generated using interpolation.

The original data can be provided upon request.

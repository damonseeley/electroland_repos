package net.electroland.gateway.mediamap;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

public class PhotosHandler extends DefaultHandler {

    @Override
    public void startElement(String uri, String localName, String qName, Attributes attributes) throws SAXException {
        super.startElement(uri, localName, qName, attributes);
        if ("entry".equalsIgnoreCase(qName)){
            String first = attributes.getValue("first");
            String last = attributes.getValue("last");
            String disambiguator = attributes.getValue("disambiguator");
            String filename = attributes.getValue("filname");
        }
    }
}
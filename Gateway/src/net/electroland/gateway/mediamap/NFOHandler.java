package net.electroland.gateway.mediamap;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

public class NFOHandler extends DefaultHandler {

    @Override
    public void startElement(String uri, String localName, String qName, Attributes attributes) throws SAXException {
        super.startElement(uri, localName, qName, attributes);
        if ("Info".equals(qName) && "name".equals(attributes.getValue("name"))){
            System.out.println(attributes.getValue("value"));
        }
    }
}
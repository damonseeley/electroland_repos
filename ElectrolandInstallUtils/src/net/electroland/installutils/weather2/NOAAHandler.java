package net.electroland.installutils.weather2;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

public class NOAAHandler extends DefaultHandler {
    public void startElement(String uri, String localName, String qName, Attributes attributes)
    throws SAXException {
        if(qName.equals("visibility_mi")) {
        } else if (qName.equals("temp_f")) {
        }
    }

}
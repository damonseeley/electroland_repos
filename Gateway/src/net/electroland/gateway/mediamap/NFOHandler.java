package net.electroland.gateway.mediamap;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

public class NFOHandler extends DefaultHandler {

    String srcFilename;
    String guid;

    @Override
    public void startElement(String uri, String localName, String qName, Attributes attributes) throws SAXException {
        super.startElement(uri, localName, qName, attributes);
        if ("Info".equalsIgnoreCase(qName) && "name".equalsIgnoreCase(attributes.getValue("name"))){
            srcFilename = attributes.getValue("value");
        }else if ("DefaultVirtualMedia".equalsIgnoreCase(qName)){
            guid = attributes.getValue("guid");
        }
    }
}
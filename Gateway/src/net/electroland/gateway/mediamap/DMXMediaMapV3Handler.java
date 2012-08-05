package net.electroland.gateway.mediamap;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

public class DMXMediaMapV3Handler extends DefaultHandler {

    final static String NULL_GUID = "00000000-0000-0000-0000-000000000000";
    
    @Override
    public void startElement(String uri, String localName, String qName, Attributes attributes) throws SAXException {
        super.startElement(uri, localName, qName, attributes);
        if ("Entry".equals(qName)){
            String idx     = attributes.getValue("idx");
            String mediaID = attributes.getValue("mediaID");
            String guid    = mediaID.substring(0, mediaID.indexOf(':'));
            String media   = attributes.getValue("media");
            if (!NULL_GUID.equals(guid)){
                System.out.println(idx + " " + media);
            }
        }
    }
}
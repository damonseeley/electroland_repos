package net.electroland.gateway.mediamap;

import java.util.HashMap;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

public class DMXMediaMapV3Handler extends DefaultHandler {

    final static String NULL_GUID = "00000000-0000-0000-0000-000000000000";
    HashMap<String, Integer> guidsToIDXs = new HashMap<String, Integer>();

    @Override
    public void startElement(String uri, String localName, String qName, Attributes attributes) throws SAXException {
        super.startElement(uri, localName, qName, attributes);
        if ("Entry".equalsIgnoreCase(qName)){
            String idxStr  = attributes.getValue("idx");
            // TODO: handle parse exception
            Integer idx    = Integer.parseInt(idxStr);
            String mediaID = attributes.getValue("mediaID");
            String guid    = mediaID.substring(0, mediaID.indexOf(':')).toLowerCase();
            if (!NULL_GUID.equals(guid)){
                guidsToIDXs.put(guid, idx);
            }
        }
    }
}
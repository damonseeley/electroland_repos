package net.electroland.gateway.mediamap;

import java.io.File;
import java.io.IOException;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;

import org.xml.sax.SAXException;

public class GenerateDB {

    public static void main(String args[]){
        SAXParserFactory factory = SAXParserFactory.newInstance();
        SAXParser saxParser;
        try {
            saxParser = factory.newSAXParser();
            // required property is location of DMXMediaMap.v3.xml and .nfo files
            saxParser.parse(new File("/Users/bradley/Documents/Electroland/Gateway/test/DMXMediaMap.v3.xml"), new DMXMediaMapV3Handler());
        } catch (ParserConfigurationException e) {
            e.printStackTrace();
        } catch (SAXException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
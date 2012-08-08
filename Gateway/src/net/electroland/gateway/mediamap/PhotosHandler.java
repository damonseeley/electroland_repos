package net.electroland.gateway.mediamap;

import java.util.ArrayList;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

public class PhotosHandler extends DefaultHandler {

    protected ArrayList<StudentMedia> students = new ArrayList<StudentMedia>();

    @Override
    public void startElement(String uri, String localName, String qName, Attributes attributes) throws SAXException {
        super.startElement(uri, localName, qName, attributes);
        if ("entry".equalsIgnoreCase(qName)){

            StudentMedia student = new StudentMedia();

            student.firstname = attributes.getValue("first").trim();
            student.lastname = attributes.getValue("last").trim();
            student.disambiguator = attributes.getValue("disambiguator").trim();
            student.srcfilename = attributes.getValue("filename").trim();
            String dateStr = attributes.getValue("createDate").trim();
            if (dateStr != null && dateStr.length() != 0){
                student.createDate = new Long(attributes.getValue("createDate"));
            }
            students.add(student);
        }
    }
}
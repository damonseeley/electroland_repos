package net.electroland.gateway.mediamap;

import java.util.ArrayList;
import java.util.Date;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

public class PhotosHandler extends DefaultHandler {

    ArrayList<StudentMedia> students = new ArrayList<StudentMedia>();
    
    @SuppressWarnings("deprecation")
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
            Date createDate = null;
            if (dateStr != null && dateStr.length() != 0){
                student.createDate = new Date(attributes.getValue("createDate"));
            }
            students.add(student);
        }
    }
}
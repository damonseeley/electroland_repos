package net.electroland.gateway.mediamap;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;

import org.apache.commons.io.FileUtils;
import org.xml.sax.SAXException;

public class GenerateDB {

    /**
     * This thing wakes up at a set period, and checks to see if the DMXMedia
     * Map file is updated.  If so, it re-renders a hash of usernames to
     * media IDs.
     * 
     * It dumps that hash to disk (JSON) and also holds it in memory.  The
     * in memory version is used to control the media player.  The disk version
     * by the web server.
     * 
     * Once its in memory, run screen saver mode and respond to requests to 
     * play videos
     * 
     * @param args
     */
    // required propertise are 
    // * location of DMXMediaMap.v3.xml and 
    // * location .nfo files
    // * minimum waits between file refreshes
    // * file refresh poll period

    // TODO: thread to resync periodically.
    // TODO: thread to listen for website requests and play them.
    final public static int fps = 33;
    final public static int clipLengthSecs = 2;

    public static void main(String args[]){

        List<StudentMedia> students
            = getRawStudentData("/Users/bradley/Documents/Electroland/Gateway/test/photos.xml");

        Map<String, String> srcFilenameToGuids 
            = mapSrcFilenameToGuids("/Users/bradley/Documents/Electroland/Gateway/test/");

        Map<String, String> guidsToIDXs
            = mapGuidsToIDXs("/Users/bradley/Documents/Electroland/Gateway/test/DMXMediaMap.v3.xml");

        students = applyGuidsToStudents(students, srcFilenameToGuids);
        students = applyIdxsToStudents(students, guidsToIDXs);

        HashMap<String, StudentMedia> studentMedia = createIdxToStudentMap(students);

        // TODO: convert studentMedia to JSON and send to webserver

        startScreenSaver(studentMedia);
    }


    private static void startScreenSaver(HashMap<String, StudentMedia>studentMedia){
        while(true){
            for (String idx : studentMedia.keySet()){
                new PlayThread(idx).start();
                try {
                    Thread.sleep(fps * clipLengthSecs * 1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }
    }


    private static HashMap<String, StudentMedia> createIdxToStudentMap(List<StudentMedia> students){

        HashMap<String, StudentMedia> idxToStudents = new HashMap<String, StudentMedia>();
        for (StudentMedia student : students){
            idxToStudents.put(student.idx, student);
        }
        return idxToStudents;
    }


    public static List<StudentMedia> applyIdxsToStudents(List<StudentMedia> students, Map<String, String> guidsToIDXs){

        for (StudentMedia student : students){
            student.idx = guidsToIDXs.get(student.guid);
        }
        return students;
    }


    public static List<StudentMedia> applyGuidsToStudents(List<StudentMedia> students, Map<String, String> srcFilenameToGuids){

        for (StudentMedia student : students){
            student.guid = srcFilenameToGuids.get(student.srcfilename);
        }
        return students;
    }


    public static Map<String, String> mapGuidsToIDXs(String DMXfilename){

        SAXParserFactory factory = SAXParserFactory.newInstance();
        DMXMediaMapV3Handler handler = new DMXMediaMapV3Handler();

        try {
            SAXParser saxParser = factory.newSAXParser();
            saxParser.parse(new File(DMXfilename), handler);
            return handler.guidsToIDXs;
        } catch (ParserConfigurationException e) {
            e.printStackTrace();
            return Collections.emptyMap();
        } catch (SAXException e) {
            e.printStackTrace();
            return Collections.emptyMap();
        } catch (IOException e) {
            e.printStackTrace();
            return Collections.emptyMap();
        }
    }


    public static Map<String, String> mapSrcFilenameToGuids(String mediaDirectory){

        HashMap<String, String> srcToMediaFilesnames = new HashMap<String, String>();
        Iterator<File> i = FileUtils.iterateFiles(new File(mediaDirectory), new String[]{".nfo"} , true);

        SAXParserFactory factory = SAXParserFactory.newInstance();
        NFOHandler handler = new NFOHandler();

        try {

            SAXParser saxParser = factory.newSAXParser();

            while (i.hasNext()){
                    File nfoFile = i.next();
                    saxParser.parse(nfoFile, handler);
                    srcToMediaFilesnames.put(handler.srcFilename, 
                                             handler.guid);
            }
        } catch (ParserConfigurationException e) {
            e.printStackTrace();
            return Collections.emptyMap();
        } catch (SAXException e) {
            e.printStackTrace();
            return Collections.emptyMap();
        } catch (IOException e) {
            e.printStackTrace();
            return Collections.emptyMap();
        }
        return srcToMediaFilesnames;
    }


    public static List<StudentMedia> getRawStudentData(String photoboothXMLfilename){

        SAXParserFactory factory = SAXParserFactory.newInstance();
        PhotosHandler handler = new PhotosHandler();

        try {
            SAXParser saxParser = factory.newSAXParser();
            saxParser.parse(new File(photoboothXMLfilename), handler);
            return handler.students;
        } catch (ParserConfigurationException e) {
            e.printStackTrace();
            return Collections.emptyList();
        } catch (SAXException e) {
            e.printStackTrace();
            return Collections.emptyList();
        } catch (IOException e) {
            e.printStackTrace();
            return Collections.emptyList();
        }
    }
}
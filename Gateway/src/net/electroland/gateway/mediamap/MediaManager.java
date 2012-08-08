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

import net.electroland.ea.EasingFunction;
import net.electroland.ea.easing.Linear;
import netP5.NetAddress;

import org.apache.commons.io.FileUtils;
import org.apache.log4j.Logger;
import org.xml.sax.SAXException;

import oscP5.OscEventListener;
import oscP5.OscMessage;
import oscP5.OscP5;
import oscP5.OscProperties;
import oscP5.OscStatus;

// TODO: properties reader
// TODO: thread to resync periodically.
// TODO: thread to listen for website requests and play them.
// TODO: integration tests with nulls
public class MediaManager implements OscEventListener {

    static Logger logger = Logger.getLogger(MediaManager.class);

    final public static int    FPS = 33;
    final public static int    CLIP_LENGTH_SECS = 2;
    final public static EasingFunction EASING_F = new Linear();

    final public static int    DB_SYNC_PERIOD_SECS = 10;
    final public static String PHOTO_XML_FILENAME = "/Users/bradley/Documents/Electroland/Gateway/test/photos.xml";
    final public static String NFO_FILES_DIR = "/Users/bradley/Documents/Electroland/Gateway/test/";
    final public static String DMX_MEDIA_MAP_FILENAME = "/Users/bradley/Documents/Electroland/Gateway/test/DMXMediaMap.v3.xml";

    final public static String PLAYER_IP = "127.0.0.1"; 
    final public static int    PLAYER_SEND_PORT = 12000;
    final public static int    PLAYER_RECEIVE_PORT = 12000;
    final public static String SET_MEDIA = "/play";
    final public static String SET_ALPHA = "/alpha";

    private OscP5 oscP5;
    private HashMap<Integer, StudentMedia> studentMedia;


    public MediaManager(){

        // TODO: these should be behind some kind of OSC abstraction
        oscP5 = configureOSC(this);
        Runtime.getRuntime().addShutdownHook(new DisposeOscClient(oscP5));

        studentMedia = syncStudentMediaFiles();
        logger.info(getStudentsJSON(studentMedia));
    }


    public static void main(String args[]){

        MediaManager mmgr = new MediaManager();

// TODO: mmgr.startResyncThread(long delay);
// TODO: mmgr.startPersonalRequestListener(int port);
        mmgr.startScreenSaver();
    }


    public static void sendStudentMediaFile(){
        // TODO: implement as PUT?
    }


    private static HashMap<Integer, StudentMedia> syncStudentMediaFiles(){
        List<StudentMedia> students
            = getRawStudentData(PHOTO_XML_FILENAME);

        Map<String, String> srcFilenameToGuids 
            = mapSrcFilenameToGuids(NFO_FILES_DIR);

        Map<String, Integer> guidsToIDXs
            = mapGuidsToIDXs(DMX_MEDIA_MAP_FILENAME);

        students = applyGuidsToStudents(students, srcFilenameToGuids);
        students = applyIdxsToStudents(students, guidsToIDXs);

        return createIdxToStudentMap(students);
    }


    private static OscP5 configureOSC(OscEventListener listener){

        OscProperties props = new OscProperties(listener);
        props.setListeningPort(PLAYER_RECEIVE_PORT);
        props.setRemoteAddress(new NetAddress(PLAYER_IP, PLAYER_SEND_PORT));
        props.setSRSP(OscProperties.ON);
        props.setDatagramSize(1024);
        props.setNetworkProtocol(OscProperties.UDP);

        System.err.println("You'll see a \"Register Dispose\" and NullPointerException");
        System.err.println("here.  Those are safely ignorable.  It's just OscP5 letting");
        System.err.println("the (non-existent) PApplet know to clean up on shutdown.");
        System.err.println();
        System.err.println("According to http://code.google.com/p/oscp5/issues/detail?id=1");
        System.err.println("this was fixed in version 0.9.8, but that doesn't jive with");
        System.err.println("their release docs or codebase.");
        System.err.println();

        return new OscP5(listener, props);
    }

    private static String getStudentsJSON(HashMap<Integer, StudentMedia>studentMedia){
        StringBuffer sb = new StringBuffer();
        sb.append('{');
        for (Integer key : studentMedia.keySet()){
            sb.append(key);
            sb.append(':');
            sb.append(studentMedia.get(key).toJSON()).append(',');
        }
        if (sb.charAt(sb.length() - 1) == ','){
            sb.setLength(sb.length() - 1);
        }
        sb.append('}');
        return sb.toString();
    }


    // TODO: proper randomization routine
    // TODO: place nice with periodic resyncs and manual requests.
    private void startScreenSaver(){
        while(true){
            for (Integer idx : studentMedia.keySet()){
                new PlayThread(idx, this).start();
                try {
                    Thread.sleep(1000 * CLIP_LENGTH_SECS);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }
    }


    private static HashMap<Integer, StudentMedia> createIdxToStudentMap(List<StudentMedia> students){

        HashMap<Integer, StudentMedia> idxToStudents = new HashMap<Integer, StudentMedia>();
        for (StudentMedia student : students){
            idxToStudents.put(student.idx, student);
        }
        return idxToStudents;
    }


    public static List<StudentMedia> applyIdxsToStudents(List<StudentMedia> students, Map<String, Integer> guidsToIDXs){

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


    public static Map<String, Integer> mapGuidsToIDXs(String DMXfilename){

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
        Iterator<File> i = FileUtils.iterateFiles(new File(mediaDirectory), new String[]{"nfo"} , true);

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


    public void setMedia(int idx){
        OscMessage mssg = new OscMessage(SET_MEDIA);
        mssg.add(idx);
        oscP5.send(mssg);
        logger.info("set media to " + idx);
    }


    public void setAlpha(float alpha){
        OscMessage mssg = new OscMessage(SET_MEDIA);
        mssg.add(alpha);
        oscP5.send(mssg);
        logger.debug("  set alpha to " + alpha);
    }


    @Override
    public void oscEvent(OscMessage mssg) {
        logger.debug(mssg);
    }


    @Override
    public void oscStatus(OscStatus mssg) {
        logger.debug(mssg);
    }
}


class DisposeOscClient extends Thread {

    private OscP5 oscP5;

    public DisposeOscClient(OscP5 oscP5){
        this.oscP5 = oscP5;
    }

    public void run(){
        oscP5.disconnect(oscP5.properties().remoteAddress());
        oscP5.dispose();
    }
 }
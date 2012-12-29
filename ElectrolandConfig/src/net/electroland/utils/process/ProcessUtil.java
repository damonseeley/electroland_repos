package net.electroland.utils.process;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ProcessUtil {

    private static int OS = 0;
    public static final int WINDOWS = 2;
    public static final int OSX = 1;

    /**
     * @param args
     */
    public static void main(String[] args) {
        // basic unit tests
        System.out.println(getRunning());
        System.out.println(run("c:\\Users\\Electroland\\Desktop\\ElVis\\run.bat", null, 1000)); // this process should stay alive until you kill the notepad instance.
    }

    public static ProcessItem run(String command, ProcessExitedListener listener, long pollPeriod){
        
        // OS check
        detectOS();

        // if windows, get the process list
        switch (OS){

        case(WINDOWS):
            List<ProcessItem> before = ProcessUtil.getRunning();
            try {
                Runtime.getRuntime().exec(command).getInputStream().close();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            List<ProcessItem> after = ProcessUtil.getRunning();
            ProcessItem newItem = getLatestWindowsJavaPID(before, after);
            new WindowsProcessPollThread(newItem, listener, pollPeriod).start();
            return newItem;

        case(OSX):
            // TODO: implement
            return null;

        default:
            System.err.println("unsupported OS.");
            return null;
        }
    }

    private static ProcessItem getLatestWindowsJavaPID(List<ProcessItem> older, List<ProcessItem> newer){
        // pretty imperfect algorithm.
        for (ProcessItem item : newer){
            if (!older.contains(item) && item.getName().equals("java.exe")){
                return item;
            }
        }
        return null;
    }
    
    public static List<ProcessItem> getRunning() {

        detectOS();

        switch (OS){
            case(OSX):
                return parseOSXList(getProcessOutput(runOSXps()));
            case(WINDOWS):
                return parseWindowsList(getProcessOutput(runWindowsTaskList()));
            default:
                System.err.println("empty Process list returned due to unsupported OS.");
                return Collections.emptyList();
        }
    }

    public static void detectOS(){
        if (OS == 0){
            String osStr = System.getProperty("os.name").toLowerCase();
            if (osStr.indexOf("mac") == 0){
                OS = OSX;
            } else if (osStr.indexOf("win") == 0){
                OS = WINDOWS;
            } else {
                System.err.println("ProcessList.java is not compatible with " + osStr);
            }
        }
    }

    private static Process runWindowsTaskList() {
        try {
            return Runtime.getRuntime().exec(System.getenv("windir") + 
                                                "\\system32\\tasklist.exe");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    private static List<ProcessItem> parseWindowsList(List<String> rawList){
        ArrayList<ProcessItem> parsedList = new ArrayList<ProcessItem>();
        int row = 0;
        for (String string : rawList){
            if (row++ > 2){ // don't parse headers
                parsedList.add(new WindowsProcessItem(string));
            }
        }
        return parsedList;
    }

    private static Process runOSXps() {
        try {
            return Runtime.getRuntime().exec("ps -e");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static List<ProcessItem> parseOSXList(List<String> rawList){
        ArrayList<ProcessItem> parsedList = new ArrayList<ProcessItem>();
        for (String string : rawList){
            parsedList.add(new OSXProcessItem(string));
        }
        return parsedList;
    }

    public static List<String> getProcessOutput(Process process) {

        ArrayList<String> list = new ArrayList<String>();

        String line;
        BufferedReader input =
                new BufferedReader(
                        new InputStreamReader(process.getInputStream()));
        try {
            while ((line = input.readLine()) != null) {
                list.add(line);
            }
            input.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return list;
    }
}
package net.electroland.utils.process;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ProcessUtil implements ProcessExitedListener{

    private static int OS = 0;
    public static final int WINDOWS = 2;
    public static final int OSX = 1;

    /**
     * @param args
     */
    public static void main(String[] args) {
        // basic unit tests

        String command = "java -classpath ./depends;./libraries/javacpp.jar;" +
        		"./libraries/javacv-windows-x86_64.jar;" +
        		"./libraries/javacv-windows-x86.jar;" +
        		"./libraries/javacv.jar;./libraries/JMyron.jar;" +
        		"./libraries/log4j-1.2.9.jar;" +
        		"./libraries/miglayout15-swing.jar;" +
        		"ELVIS.jar; net.electroland.elvis.blobktracking.core.ElVisServer";
        
        // in practice, read this from a .bat file.
        File runDir = new File("C:\\Users\\Electroland\\Desktop\\Elvis");

        System.out.println(run(command, runDir, new ProcessUtil(), 1000)); // this process should stay alive until you kill the notepad instance.
    }

    /**
     * windows version: we're only detecting java.exe.
     */
    public static ProcessItem run(String command, File runDirectory, ProcessExitedListener listener, long pollPeriod){
        
        // OS check
        detectOS();

        // if windows, get the process list
        switch (OS){

        case(WINDOWS):
            List<ProcessItem> before = ProcessUtil.getRunning();
            try {
                Runtime.getRuntime().exec(command, null, runDirectory).getInputStream().close();
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
        ArrayList<ProcessItem> newItems = new ArrayList<ProcessItem>();
        // pretty imperfect algorithm.
        for (ProcessItem item : newer){
            if (!older.contains(item) 
             && !item.getName().equals("tasklist.exe")  // launched to get this list
             && !item.getName().equals("conhost.exe")   // launched for security
             && !item.getName().equals("cmd.exe")){     // launched to support this process
                newItems.add(item);
            }
        }
        return newItems.get(0);
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

    @Override
    public void exited(ProcessItem ded) {
        System.out.println("EXIT: " + ded);
        String command = "java -classpath ./depends;./libraries/javacpp.jar;" +
                "./libraries/javacv-windows-x86_64.jar;" +
                "./libraries/javacv-windows-x86.jar;" +
                "./libraries/javacv.jar;./libraries/JMyron.jar;" +
                "./libraries/log4j-1.2.9.jar;" +
                "./libraries/miglayout15-swing.jar;" +
                "ELVIS.jar; net.electroland.elvis.blobktracking.core.ElVisServer";
        
        // in practice, read this from a .bat file.
        File runDir = new File("C:\\Users\\Electroland\\Desktop\\Elvis");
        
        System.out.println(run(command, runDir, new ProcessUtil(), 1000)); // this process should stay alive until you kill the notepad instance.
    }
}
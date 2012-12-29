package net.electroland.utils.process;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

public class ProcessList {

    private static int OS = 0;
    public static final int WINDOWS = 2;
    public static final int OSX = 1;

    /**
     * @param args
     */
    public static void main(String[] args) {
        // basic unit tests
        System.out.println(getRunning());
        List<ProcessItem> javas = getRunningInstancesOf("javaw.exe");
        System.out.println(javas);
        System.out.println(isRunning(javas.get(0)));
    }


    public static List<ProcessItem> getRunningInstancesOf(String name){

        List<ProcessItem> all = getRunning();
        Iterator<ProcessItem> i = all.iterator();

        while (i.hasNext()){
            ProcessItem next = i.next();
            System.out.println("comparing: " + next);
            if (!next.getName().equals(name)){
                i.remove();
            }
        }

        return all;
    }

    public static boolean isRunning(ProcessItem check) {
        List<ProcessItem> all = getRunning();
        for (ProcessItem item : all){
            if (check.equals(item)){
                return true;
            }
        }
        return false;
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
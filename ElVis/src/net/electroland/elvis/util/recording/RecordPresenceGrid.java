package net.electroland.elvis.util.recording;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.net.SocketException;

import net.electroland.elvis.net.GridData;
import net.electroland.elvis.net.PresenceGridUDPClient;

public class RecordPresenceGrid extends PresenceGridUDPClient {

    private FileRecorder recorder;

    public RecordPresenceGrid(int port) throws SocketException {
        super(port);
    }

    public static void main(String args[]) throws Exception {
        if (args.length != 2){
            System.out.println("Usage: RecordPresenceGrid [port] [filename]");
        }else{
            RecordPresenceGrid r = new RecordPresenceGrid(new Integer(args[0]));
            r.recorder = new FileRecorder(args[1]);
        }
    }

    @Override
    public void handle(GridData t) {
        if (recorder != null){
            try {
                recorder.recored(t);
            } catch (IOException e) {
                e.printStackTrace();
                recorder.close();
            }
        }
    }
}
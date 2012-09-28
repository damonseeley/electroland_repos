package net.electroland.elvis.util.recording;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.net.SocketException;

import net.electroland.elvis.blobtracking.BaseTrack;
import net.electroland.elvis.blobtracking.TrackResults;
import net.electroland.elvis.net.GridData;
import net.electroland.elvis.net.TrackUDPClient;

public class RecordTracks extends TrackUDPClient {

    private FileRecorder recorder;

    public RecordTracks(int port) throws SocketException {
        super(port);
    }

    public static void main(String args[]) throws Exception {
        if (args.length != 2){
            System.out.println("Usage: RecordPresenceGrid [port] [filename]");
        }else{
            RecordTracks r = new RecordTracks(new Integer(args[0]));
            r.recorder = new FileRecorder(args[1]);
        }
    }

    @Override
    public void handle(TrackResults<BaseTrack> t) {
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
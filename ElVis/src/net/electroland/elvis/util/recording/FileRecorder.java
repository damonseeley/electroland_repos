package net.electroland.elvis.util.recording;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import net.electroland.elvis.net.StringAppender;

public class FileRecorder implements Recorder {

    private BufferedWriter output;
    private StringBuilder  buffer;

    public FileRecorder(String filename, String header) throws IOException {
        open(filename);
        recordHeader(header);
    }

    @Override
    public void open(String filename) throws IOException {
        if (output != null){
            close();
        }
        output = new BufferedWriter(new FileWriter(new File(filename)));
        buffer = new StringBuilder();
    }

    @Override
    public void recordHeader(String header) throws IOException {
        output.write(header);
        output.newLine();
        output.flush();
    }

    @Override
    public void record(StringAppender a) throws IOException {

        buffer.setLength(0);
        buffer.append(System.currentTimeMillis());
        buffer.append(':');

        a.buildString(buffer);

        output.write(buffer.toString());
        output.newLine();
        output.flush();
    }

    @Override
    public void close() {
        try {
            output.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
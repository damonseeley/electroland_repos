package net.electroland.elvis.util.recording;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import net.electroland.elvis.net.StringAppender;

public class FileRecorder {

    private BufferedWriter output;
    private StringBuilder buffer;

    public FileRecorder(String filename) throws IOException {
        output = new BufferedWriter(new FileWriter(new File(filename)));
        buffer = new StringBuilder();
    }

    public void recored(StringAppender a) throws IOException {
        if (output != null){

            buffer.setLength(0);
            buffer.append(System.currentTimeMillis());
            buffer.append(':');

            a.buildString(buffer);

            output.write(buffer.toString());
            output.newLine();
            output.flush();
        }
    }

    public void close() {
        try {
            output.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

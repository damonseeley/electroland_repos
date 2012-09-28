package net.electroland.elvis.util.recording;

import java.io.IOException;

import net.electroland.elvis.net.StringAppender;


public interface Recorder {
    public void open(String filename) throws IOException;
    public void recordHeader(String header) throws IOException;
    public void recordLine(StringAppender a) throws IOException;
    public void close();
}
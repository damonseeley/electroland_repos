package net.electroland.gotham.processing;

import java.io.File;
import java.util.List;

import net.electroland.elvis.blobtracking.BaseTrack;
import net.electroland.elvis.net.GridData;
import net.electroland.utils.lighting.canvas.ELUPApplet;

import org.apache.log4j.Logger;

import processing.event.KeyEvent;

abstract public class GothamPApplet extends ELUPApplet {

    static Logger logger = Logger.getLogger(GothamPApplet.class);
    private static final long serialVersionUID = 8991292715035010606L;

    @Override
    public void handleKeyEvent(KeyEvent e){
        logger.debug("Ignoring key events in GothamPApplet.");
    }

    public void handle(GridData d) {}

    public void handle(List<BaseTrack> tr) {}

    // not going to make abstract because this is really just for MovieOrImagePApplet or some alternative Michael creates
    public void fileReceived(File filename){
        // override me
    }
}
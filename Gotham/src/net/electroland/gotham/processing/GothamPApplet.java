package net.electroland.gotham.processing;

import java.util.List;
import java.io.File;
import java.util.Collection;

import net.electroland.elvis.blobtracking.BaseTrack;
import net.electroland.elvis.net.GridData;
import net.electroland.gotham.core.Person;
import net.electroland.gotham.core.Room;
import net.electroland.utils.lighting.canvas.ELUPApplet;

import org.apache.log4j.Logger;

import processing.event.KeyEvent;

abstract public class GothamPApplet extends ELUPApplet {

    static Logger logger = Logger.getLogger(GothamPApplet.class);
    private static final long serialVersionUID = 8991292715035010606L;
    private Room room;

    @Override
    public void handleKeyEvent(KeyEvent e){
        logger.debug("Ignoring key events in GothamPApplet.");
    }

    // TODO: should probably make this abstract, but a bunch of subclasses have not implemented yet.
    public void handle(GridData d) {}

    // TODO: should probably make this abstract, but a bunch of subclasses have not implemented yet.
    public void handle(List<BaseTrack> tr) {}

    public Collection<Person> getPersons(){
        return room.getPersons();
    }

    // not going to make abstract because this is really just for MovieOrImagePApplet or some alternative Michael creates
    public void fileReceived(File filename){
        // override me
    }
}
package net.electroland.gotham.processing;

import java.io.File;
import java.util.Collection;

import net.electroland.elvis.net.GridData;
import net.electroland.gotham.core.Person;
import net.electroland.gotham.core.Room;
import net.electroland.utils.lighting.canvas.ELUPApplet;

abstract public class GothamPApplet extends ELUPApplet {

    private static final long serialVersionUID = 1L;
    private Room room;

    public void handle(GridData d) {
        synchronized(room){
            room = new Room(d);
        }
    }

    public Collection<Person> getPersons(){
        return room.getPersons();
    }

    // not going to make abstract because this is really just for MovieOrImagePApplet or some alternative Michael creates
    public void fileReceived(File filename){
        // override me
    }
}
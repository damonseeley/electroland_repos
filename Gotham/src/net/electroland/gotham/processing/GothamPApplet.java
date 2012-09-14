package net.electroland.gotham.processing;

import java.io.File;

import net.electroland.elvis.net.GridData;
import net.electroland.gotham.core.People;
import net.electroland.utils.lighting.canvas.ELUPApplet;

abstract public class GothamPApplet extends ELUPApplet {

    private static final long serialVersionUID = 1L;
    private People pm;

    public void handle(GridData d) {
        synchronized(pm){
            pm = new People(d);
        }
    }

    public People getPeople(){
        return pm;
    }

    // not going to make abstract because this is really just for MovieOrImagePApplet or some alternative Michael creates
    public void fileReceived(File filename){
        // override me
    }
}
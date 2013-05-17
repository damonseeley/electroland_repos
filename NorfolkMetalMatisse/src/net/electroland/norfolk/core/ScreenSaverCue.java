package net.electroland.norfolk.core;

import net.electroland.utils.ParameterMap;

public class ScreenSaverCue extends Cue {

    private boolean isSaving = false;
    private int timeout, fadeout;
    private Cue[] exceptions;

    public ScreenSaverCue(ParameterMap p) {
        super(p);
        timeout = p.getRequiredInt("timeout");
        fadeout = p.getRequiredInt("fadeout");
    }

    @Override
    public void fire(EventMetaData meta, ClipPlayer cp) {
        if (isSaving){
            cp.enterScreensaverMode(fadeout);
        }else{
            cp.exitScreensaverMode(fadeout);
        }
    }

    @Override
    public boolean ready(EventMetaData meta) {

        boolean everythingInactive = meta.getTimeSinceLastCueExcluding(exceptions) > timeout;

        if (everythingInactive && !isSaving){
            isSaving = true;
            return true;
        }else if (!everythingInactive && isSaving){
            isSaving = false;
            return true;
        }else{
            return false;
        }
    }

    public void setExceptions(Cue... exceptions){
        this.exceptions = exceptions;
    }
}
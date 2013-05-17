package net.electroland.norfolk.core;

import java.util.List;
import java.util.Random;

import org.apache.log4j.Logger;

import net.electroland.utils.ParameterMap;

public class BigShowCue extends Cue {

    private static Logger logger = Logger.getLogger(BigShowCue.class);
    private int waitMillis;
    private List<String>cues;
    private String trainChannelId;

    public BigShowCue(ParameterMap p) {
        super(p);
        waitMillis      = p.getRequiredInt("waitMillis");
        cues            = p.getRequiredList("cues");
        trainChannelId  = p.getRequired("trainChannelId");
    }

    @Override
    public void fire(EventMetaData meta, ClipPlayer cp) {
        logger.info("RUN BIG SHOW");
        cp.play(cues.get(new Random().nextInt(cues.size())));
    }

    @Override
    public boolean ready(EventMetaData meta) {
    	boolean isInactive = meta.totalSensorsEventsOverLastExcluding(60000, trainChannelId) == 0;
        return isInactive && meta.getTimeSinceLastCue(this) > waitMillis;
    }
}
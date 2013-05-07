package net.electroland.norfolk.eio.vchannels;

import net.electroland.norfolk.eio.filters.PeopleIOWatcher;
import net.electroland.norfolk.eio.filters.PersonEventDetectionFilter;
import net.electroland.norfolk.eio.filters.PersonPresenceFilter;
import net.electroland.norfolk.eio.filters.HoldoffFilter;
import net.electroland.eio.Value;
import net.electroland.eio.ValueSet;
import net.electroland.eio.VirtualChannel;
import net.electroland.utils.ParameterMap;

/**
 * This class manages all processing used to detect and classify person entrance
 * and exit events for the Norfolk IR sensors. It takes as its input channels the
 * left and right channels of an IR sensor (in that order), and outputs integer 
 * PersonEventCode values which are interpreted by PeopleIOWatcher to trigger
 * corresponding PersonEvent notifications.
 * 
 * There are only 3 parameters of this class which are made public for specification
 * in an io.properties file. They are as follows:
 * 
 *    "detectorThresh" - A value in the range [0 16383] that specifies the threshold
 *                level for the detector signal. The default value is 5200. This value
 *                should be increased if events are triggering spontaneously due to 
 *                background noise. The maximum detector value observed in previously 
 *                measured noise data was ~3400 during the day and ~400 at night.
 *                Making this value too high may cause events to go undetected and will
 *                increase the response time of the system.
 *    
 *    "detectorHoldoffLenMs" - A value specifying the minimum hold-off time for event detection
 *                in milliseconds. The default value is 1200. This value should be 
 *                increased if too many "double-tap" events are occurring. Note that this
 *                is only a minimum reset time - if continue activity is detected the system
 *                will wait until it settles before re-arming the detector.
 *    
 *    "clipThresh" - A value in the range [0 16383] that specifies the threshold level
 *                for input signals to be declared "clipped". The default value is 14900.
 *                Because the values coming from the A/D converter don't quite fill
 *                the range [-16383 16383] and because our adjustment to make the data nearly
 *                zero-mean may be imperfect, we allow for some padding by declaring smaller
 *                values to be clipped as well. This threshold is used by an alternative
 *                "clip" detector, which helps detect events that occur in rapid succession.
 * 
 * Adjusting the rest of the parameters is not advisable without performing more detailed
 * analysis of the resulting change in behavior using external tools (MATLAB).
 * 
 * 
 * @author Sean
 *
 */

public class PersonTrackerVirtualChannel extends VirtualChannel {

    // Input data is assumed to be between +- Short.MAX_VALUE / 2, (~= 2^14 == 16384)
    private final static float normFac = Short.MAX_VALUE / 2.0f;
    
    // Algorithm parameters
    private float   detectorThresh;
    private double  detectorHoldoffAdjustAfterExitEventMs;
    
    private float clipThresh;
    private float clipReArmFraction;


    // State variables
    private boolean detectorArmed, personPresent;
    private boolean[] clipDetectorArmed;
    private int prevEventMeanSignalSign;
    private int prevEventFirstClipSign;


    // Filters
    private PersonEventDetectionFilter detectionFilt;
    private HoldoffFilter detectionHoldoffFilt;
    
    private PersonPresenceFilter personPresentFilt;
    private HoldoffFilter personPresentHoldoffFilt;


    // For debug
    private final static long startTimeMs = System.currentTimeMillis();


    @Override
    public void configure(ParameterMap params) {
        
        // Threshold parameters
        detectorThresh = params.getDefaultInt("detectorThresh", 5200) / normFac;
        clipThresh = params.getDefaultInt("clipThresh", 14900) / normFac;
        
        
        // Internal parameters (not publicly set-able)
        clipReArmFraction = 0.9f;
        detectorHoldoffAdjustAfterExitEventMs = 600.0;
        
        
        // Configure detectionFilt (all params are hard-coded in class)
        ParameterMap empty = new ParameterMap();
        detectionFilt = new PersonEventDetectionFilter();
        detectionFilt.configure(empty);
        
        
        // Configure personPresentFilt (all params are hard-coded in class)
        personPresentFilt = new PersonPresenceFilter();
        personPresentFilt.configure( empty );
        
        
        // Configure hold-off filter for PersonEvent detector
        ParameterMap detectionHoldoffFiltParams = new ParameterMap();
        detectionHoldoffFiltParams.put("holdoffLenMs", Double.toString( params.getDefaultDouble("detectorHoldoffLenMs", 1200.0)) );
        detectionHoldoffFiltParams.put("maxHoldoffLenMs", Double.toString( params.getDefaultDouble("detectorMaxHoldoffLenMs", 10000.0)) );
        detectionHoldoffFiltParams.put("penaltyThresh", Double.toString(0.1*detectorThresh));
        detectionHoldoffFiltParams.put("resetThresh", Double.toString(0.5*detectorThresh));
        detectionHoldoffFiltParams.put("penaltyMult", "2.0");
        detectionHoldoffFiltParams.put("penaltyPow", "1.5");
        detectionHoldoffFilt = new HoldoffFilter();
        detectionHoldoffFilt.configure( detectionHoldoffFiltParams );
        
        
        // Configure hold-off filter for personPresentFilt
        ParameterMap personPresentHoldoffFiltParams = new ParameterMap();
        personPresentHoldoffFiltParams.put("holdoffLenMs", "9000");
        personPresentHoldoffFiltParams.put("penaltyThresh", "0.05");
        personPresentHoldoffFiltParams.put("resetThresh", "0.3");
        personPresentHoldoffFiltParams.put("penaltyMult", "4.0");
        personPresentHoldoffFiltParams.put("penaltyPow", "1.5");
        personPresentHoldoffFilt = new HoldoffFilter();
        personPresentHoldoffFilt.configure( personPresentHoldoffFiltParams );
        
        
        // Initialize state variables
        detectorArmed = true;
        clipDetectorArmed = new boolean[2];
        clipDetectorArmed[0] = true;
        clipDetectorArmed[1] = true;
        
        personPresent = false;
        prevEventMeanSignalSign = 0;
        prevEventFirstClipSign = 0;
        
    }

    @Override
    public Value processInputs(ValueSet inputValues) {
        
        // Validate input ValueSet
        if (inputValues.values().size() != 2)
            throw new RuntimeException("PersonTrackerVirtualChannel requires exactly 2 inputValues.");
        
        
        // Initialize output value to indicate no event (will be changed before returning if an event is detected)
        Value output = new Value(0);
        
        
        // Get ordered input channel values so that diffVal will always be ch1 - ch2
        int[] inputVals = new int[2];
        inputVals[0] = inputValues.get( this.getChannels().get(0) ).getValue();
        inputVals[1] = inputValues.get( this.getChannels().get(1) ).getValue();
        
        // Form mean and difference of the two inputs (divide by 2 to keep values in the range
        //   +- Short.MAX_VALUE / 2)
        Value meanVal = new Value( (inputVals[0] + inputVals[1]) / 2 );
        Value diffVal = new Value( (inputVals[0] - inputVals[1]) / 2 );
        
        
        
        // Calculate the current person presence detector value using the difference signal
        Value personPresentVal = new Value( diffVal.getValue() );
        personPresentFilt.filter(personPresentVal);
        
        // Update our no-person-present-declaration hold-off counter
        if (personPresent) {
            
            personPresentHoldoffFilt.filter( personPresentVal );
            
            // As long as the no-person-present-declaration hold-off period hasn't ended, we will
            //   continue to assume that a person is present. 
            personPresent = (personPresentVal.getValue() == 0);
            
            if (!personPresent) {
                // Make sure the next event is treated as an entrance if it is a clip event
                prevEventFirstClipSign = 0;
//                System.out.println("CH " + id + " - PERSON PRESENT TIME-OUT AT " + (((double)System.currentTimeMillis() - startTimeMs)/1000));
            }
        
        }
        
        
        
        // Calculate the current event detector value using the mean signal
        Value detectorVal = new Value( meanVal.getValue() );
        detectionFilt.filter( detectorVal );
        
        // If our detector isn't currently armed, the clip detectors can't be armed either - we won't 
        //    declare a new event using either detector. Check to see if the current event waveform has
        //    clipped positive or negative and update our event detection hold-off counter.
        if (!detectorArmed) {
            
            // If the current event waveform hasn't clipped yet, check if it has now (in either channel)
            int i = 0;
            while (prevEventFirstClipSign == 0 && i < 2) {
                if ( Math.abs( inputVals[i] / normFac ) > clipThresh )
                    prevEventFirstClipSign = (int) Math.signum(inputVals[i]);
                i++;
            }
            
            // Update the detection hold-off timer  
            detectionHoldoffFilt.filter( detectorVal );
            
            // If the event detection hold-off period has ended, we will re-arm the detector
            detectorArmed = (detectorVal.getValue() == 1);
            
//            if (detectorArmed)
//                System.out.println("CH " + id + " - DETECTOR RE-ARMED AT " + (((double)System.currentTimeMillis() - startTimeMs)/1000));
        
        }
        // Otherwise, the detector is armed, check if a new event has occurred
        else {
            
            float normalizedDetectorVal = ((float) detectorVal.getValue()) / ((float) Integer.MAX_VALUE);
            
            // Check if we've found a new event using the normal detector signal
            if ( normalizedDetectorVal > detectorThresh ) {
                
                // Check the current sign of the mean signal
                int currEventMeanSignalSign = (int) Math.signum( meanVal.getValue() );
                double detectorHoldoffAdjustMs;
                
                // If we believe that a person is currently present, we will declare this to be an exit
                //    event if the currEventMeanSignalSign is opposite the prevEventMeanSignalSign
                if (personPresent && (currEventMeanSignalSign != 0) && (currEventMeanSignalSign != prevEventMeanSignalSign)) {
                    
                    // A person has exited the sensor focus
                    if ( Math.abs(inputVals[0]) < Math.abs(inputVals[1]) ) {
                        output.setValue( PeopleIOWatcher.PersonEventCodes.EXIT_L );
                        System.out.println("CH " + id + " - PERSON EXITED TO LEFT");
                    }
                    else {
                        output.setValue( PeopleIOWatcher.PersonEventCodes.EXIT_R );
                        System.out.println("CH " + id + " - PERSON EXITED TO RIGHT");
                    }
                    
                    
                    // Update state variables
                    personPresent = false;
                    // NOTE: (prevEvent* variables won't be examined again until after being set by an 
                    //   entrance event, except possibly during a clip-detector based event. Here, we leave
                    //   those values alone so that subsequent clip-detector based events will compare clip
                    //   sign to the most "entrance"-declared event, since their classification logic (entrance/
                    //   exit event decision) assumes this.
                    
                    detectorHoldoffAdjustMs = detectorHoldoffAdjustAfterExitEventMs;
                
                }
                else {
                    
                    // Otherwise, we declare this to be an entrance event
                    if ( Math.abs(inputVals[0]) > Math.abs(inputVals[1]) ) {
                        output.setValue( PeopleIOWatcher.PersonEventCodes.ENTER_L );
                        System.out.println("CH " + id + " - PERSON ENTERED FROM LEFT" );
                    }
                    else {
                        output.setValue( PeopleIOWatcher.PersonEventCodes.ENTER_R );
                        System.out.println("CH " + id + " - PERSON ENTERED FROM RIGHT" );
                    }
                    
                    
                    // Update state variables
                    personPresent = true;
                    prevEventMeanSignalSign = currEventMeanSignalSign;
                    prevEventFirstClipSign = 0;
                    
                    detectorHoldoffAdjustMs = 0.0;
                
                }
                
                // In both cases (entrance / exit event), disarm the detectors to avoid re-triggering on 
                //    the following portion of response
                detectorArmed = false;
                clipDetectorArmed[0] = false;
                clipDetectorArmed[1] = false;
                
                // And start hold-off timers
                detectionHoldoffFilt.startHoldoff(detectorHoldoffAdjustMs);
                personPresentHoldoffFilt.startHoldoff();
                
                
                // Done handling the current detector-signal-based event
            
            }
            // Otherwise, the normal detector is armed but hasn't detected an event this sample
            //    Check if we should declare an event because one of the channels has clipped
            else {
                
                int i = 0;
                while ( i < 2 && (output.getValue() == 0) ) {
                    
                    // If the clipping-based detector isn't currently armed, check if it should be (we will re-arm it
                    //    only once the normal detector is armed and the meanSig value has fallen below some threshold
                    //    so that it doesn't trip immediately on the same clipped region of signal)
                    if (!clipDetectorArmed[i] && (Math.abs(inputVals[i] / normFac) < (clipReArmFraction * clipThresh)) ) {
                        clipDetectorArmed[i] = true;
//                        System.out.println("CH " + id + " - CLIP DETECTOR " + i + " ARMED AT " + ((double)System.currentTimeMillis() - startTimeMs)/1000);
                    }
                    // Otherwise, if this channel's clip detector is armed and the channel clipped this sample, we have 
                    //    a clip-based event detection
                    else if (clipDetectorArmed[i] && (Math.abs(inputVals[i] / normFac) > clipThresh) ) { 
                        
                        int currEventClipSign = (int) Math.signum(inputVals[i]);
                        double detectorHoldoffAdjustMs;
                        
                        if (prevEventFirstClipSign == 0 || currEventClipSign == prevEventFirstClipSign) {
                            
                            // If prevEventFirstClipSign == 0, the previous entrance event didn't clip, so we can't
                            //    assess if this is the same kind of event based on the polarity of clipping. To err
                            //    on the side of over-firing, we assume it is an entrance event .
                            // Otherwise, if this event clipped with the same polarity as the previous entrance event
                            //    we  will declare it to be an entrance event as well and fire a PersonEvent.
                            if (i == 0) {
                                output.setValue( PeopleIOWatcher.PersonEventCodes.ENTER_L );
                                System.out.println("CH " + id + " - PERSON ENTERED (BY CLIP) FROM LEFT");
                            }
                            else {
                                output.setValue( PeopleIOWatcher.PersonEventCodes.ENTER_R );
                                System.out.println("CH " + id + " - PERSON ENTERED (BY CLIP) FROM RIGHT");
                            }
                            
                            
                            // Update state variables
                            personPresent = true;
                            prevEventMeanSignalSign = (int) Math.signum( meanVal.getValue() );
                            prevEventFirstClipSign = currEventClipSign;
                            
                            detectorHoldoffAdjustMs = 0.0;
                        }
                        else {
                            
                            // The previous entrance event did clip and this event is clipping with the opposite 
                            //    polarity. Thus we assume that this is an exit event paired with that entrance.
                            if (i == 0) {
                                output.setValue( PeopleIOWatcher.PersonEventCodes.EXIT_R );
                                System.out.println("CH " + id + " - PERSON EXITED (BY CLIP) TO RIGHT");
                            }
                            else {
                                output.setValue( PeopleIOWatcher.PersonEventCodes.EXIT_L );
                                System.out.println("CH " + id + " - PERSON EXITED (BY CLIP) TO LEFT");
                            }
                            
                            // Update state variables
                            personPresent = false;
                            // (See note in normal detector exit event about prevEvent* variables)
                            
                            detectorHoldoffAdjustMs = detectorHoldoffAdjustAfterExitEventMs;
                        }
                        
                        // In both cases (entrance / exit event), disarm the detectors to avoid re-triggering on 
                        //    the following portion of response
                        detectorArmed = false;
                        clipDetectorArmed[0] = false;
                        clipDetectorArmed[1] = false;
                        
                        // And start hold-off timers
                        detectionHoldoffFilt.startHoldoff(detectorHoldoffAdjustMs);
                        personPresentHoldoffFilt.startHoldoff();
                    
                    }
                    
                    i++; // Move on to the next channel
                
                } // end for loop through ch1 and ch2
            
            } // Done checking for clip-detector based events
        
        } // Done checking for events
        
        
        return output;
    }
    
}
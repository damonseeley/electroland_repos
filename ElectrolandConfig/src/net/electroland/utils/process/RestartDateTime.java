package net.electroland.utils.process;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.Locale;

public class RestartDateTime {

    public final static String HOURLY = "hourly";
    public final static String DAILY  = "daily";
    public final static String WEEKLY = "weekly";
    
    public final static String HOURLY_FORMAT = "mm";
    public final static String DAILY_FORMAT  = "h:mm a";
    public final static String WEEKLY_FORMAT = "EEE, h:mm a";

    private String repeatRate;
    private Calendar reference;

    public RestartDateTime(String repeatRate, String repeatDayTime){

        this.repeatRate = repeatRate;
        Date referenceDate = null;

        if (isHourly()){
            referenceDate = parseHourly(repeatDayTime);
        } else if (isDaily()){
            referenceDate = parseDaily(repeatDayTime);
        } else if (isWeekly()){
            referenceDate = parseWeekly(repeatDayTime);
        } else {
            throw new RuntimeException("Unknown type: " + repeatRate);
        }

        Calendar reference = Calendar.getInstance();
        reference.setTime(referenceDate);
    }

    /**
     * set minutes for all date types
     * @return
     */
    public int getMinutes(){
        return reference.get(Calendar.MINUTE);
    }

    /**
     * set hours for daily, weekly but not hourly
     * @return
     */
    public int getHour(){
        return reference.get(Calendar.HOUR);
    }

    /**
     * set day for weekly, but not hourly or daily
     * @return
     */
    public int getDay(){
        return reference.get(Calendar.DAY_OF_WEEK);
    }

    public static Date parseHourly(String dateParam){
        try{
            return new SimpleDateFormat(HOURLY_FORMAT, Locale.ENGLISH).parse(dateParam);
        }catch(ParseException e){
            throw new RuntimeException(e);
        }
    }

    public static Date parseDaily(String dateParam){
        try{
            return new SimpleDateFormat(DAILY_FORMAT, Locale.ENGLISH).parse(dateParam);
        }catch(ParseException e){
            throw new RuntimeException(e);
        }
    }

    public static Date parseWeekly(String dateParam){
        try{
            return new SimpleDateFormat(WEEKLY_FORMAT, Locale.ENGLISH).parse(dateParam);
        }catch(ParseException e){
            throw new RuntimeException(e);
        }
    }

    public boolean isHourly(){
        return HOURLY.equals(repeatRate);
    }

    public boolean isDaily(){
        return DAILY.equals(repeatRate);
    }

    public boolean isWeekly(){
        return WEEKLY.equals(repeatRate);
    }

}
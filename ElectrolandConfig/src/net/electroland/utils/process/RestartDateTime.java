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
    
    public final static String HOURLY_FORMAT  = "mm";
    public final static String HOURLY_EXAMPLE = "05";
    public final static String DAILY_FORMAT   = "h:mm a";
    public final static String DAILY_EXAMPLE  = "12:05 PM";
    public final static String WEEKLY_FORMAT  = "EEE, h:mm a";
    public final static String WEEKLY_EXAMPLE = "Wed, 12:59 AM";

    private String repeatRate;
    private Calendar reference;

    public RestartDateTime(String repeatRate, String repeatDayTime){

        this.repeatRate = repeatRate;
        Date referenceDate = null;

        if (isHourly()){
            referenceDate = parse(HOURLY_FORMAT, repeatDayTime);
        } else if (isDaily()){
            referenceDate = parse(DAILY_FORMAT, repeatDayTime);
        } else if (isWeekly()){
            referenceDate = parse(WEEKLY_FORMAT, repeatDayTime);
        } else {
            throw new RuntimeException("Unknown type: " + repeatRate);
        }

        reference = Calendar.getInstance();
        reference.setTime(referenceDate);
    }

    // unit tests (remember -ea as a VM param)
    public static void main(String[] args){

        RestartDateTime rdt0 = new RestartDateTime(RestartDateTime.HOURLY, "04");
        assert rdt0.isHourly();
        assert rdt0.getMinutes() == 4;

        RestartDateTime rdt1 = new RestartDateTime(RestartDateTime.DAILY, "12:08 PM");
        assert rdt1.isDaily();
        assert rdt1.getHour() == 0;
        assert rdt1.getMinutes() == 8;
        assert rdt1.getAmPm() == Calendar.PM;

        RestartDateTime rdt2 = new RestartDateTime(RestartDateTime.WEEKLY, "Mon, 12:00 AM");

        assert rdt2.isWeekly();
        assert rdt2.getDay() == Calendar.MONDAY;
        assert rdt2.getHour() == 0;
        assert rdt2.getMinutes() == 00;
        assert rdt2.getAmPm() == Calendar.AM;
    }

    public int getMinutes(){
        return reference.get(Calendar.MINUTE);
    }

    public int getHour(){
        return reference.get(Calendar.HOUR);
    }

    public int getAmPm(){
        return reference.get(Calendar.AM_PM);
    }

    public int getDay(){
        return reference.get(Calendar.DAY_OF_WEEK);
    }

    public static Date parse(String format, String dateParam){
        try{
            SimpleDateFormat sdt = new SimpleDateFormat(format, Locale.ENGLISH);
            sdt.setLenient(false);
            return sdt.parse(dateParam);
        }catch(ParseException e){
            System.err.println("The following date formats are supported:");
            System.err.println("  " + HOURLY + ":\t" + HOURLY_FORMAT + "\tExample: " + HOURLY_EXAMPLE);
            System.err.println("  " + DAILY  + ":\t" + DAILY_FORMAT  + "\tExample: " + DAILY_EXAMPLE);
            System.err.println("  " + WEEKLY + ":\t" + WEEKLY_FORMAT + "\tExample: " + WEEKLY_EXAMPLE);
            System.err.println();
            throw new RuntimeException(e);
        }
    }

    public boolean isHourly(){
        return HOURLY.equalsIgnoreCase(repeatRate);
    }

    public boolean isDaily(){
        return DAILY.equalsIgnoreCase(repeatRate);
    }

    public boolean isWeekly(){
        return WEEKLY.equalsIgnoreCase(repeatRate);
    }
}
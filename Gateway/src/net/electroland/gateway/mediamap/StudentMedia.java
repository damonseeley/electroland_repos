package net.electroland.gateway.mediamap;


public class StudentMedia {
    String firstname, lastname, disambiguator;
    String idx, guid, srcfilename;
    Long createDate;

    public String toString(){
        StringBuffer sb = new StringBuffer();
        sb.append("{");
        sb.append("\"firstname\":").append('"').append(firstname).append('"');
        sb.append(',');
        sb.append("\"lastname\":").append('"').append(lastname).append('"');
        sb.append(',');
        sb.append("\"disambiguator\":").append('"').append(disambiguator).append('"');
        sb.append(',');
        sb.append("\"idx\":").append('"').append(idx).append('"');
        sb.append(',');
        sb.append("\"guid\":").append('"').append(guid).append('"');
        sb.append(',');
        sb.append("\"srcfilename\":").append('"').append(srcfilename).append('"');
        sb.append("}");
        return sb.toString();
    }
}
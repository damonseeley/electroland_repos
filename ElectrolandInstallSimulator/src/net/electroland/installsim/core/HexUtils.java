package net.electroland.installsim.core;
 
public class HexUtils {
	
    public static void printHex(byte[] b) {
        for (int i = 0; i < b.length; ++i) {
            if (i % 16 == 0) {
                System.out.print (Integer.toHexString ((i & 0xFFFF) | 0x10000).substring(1,5) + " - ");
            }
            System.out.print (Integer.toHexString((b[i]&0xFF) | 0x100).substring(1,3) + " ");
            if (i % 16 == 15 || i == b.length - 1)
            {
                int j;
                for (j = 16 - i % 16; j > 1; --j)
                    System.out.print ("   ");
                System.out.print (" - ");
                int start = (i / 16) * 16;
                int end = (b.length < i + 1) ? b.length : (i + 1);
                for (j = start; j < end; ++j)
                    if (b[j] >= 32 && b[j] <= 126)
                        System.out.print ((char)b[j]);
                    else
                        System.out.print (".");
                System.out.println ();
            }
        }
        System.out.println();
    }
    
    public static String bytesToHex(byte[] b, int newByteLength) {
    	String hexString = "";
        //for (int i = 0; i < b.length; ++i) {
        for (int i = 0; i < newByteLength; ++i) {
//            if (i % 16 == 0) {
//                System.out.print (Integer.toHexString ((i & 0xFFFF) | 0x10000).substring(1,5) + " - ");
//            }
            //System.out.print (Integer.toHexString((b[i]&0xFF) | 0x100).substring(1,3) + " ");
            hexString += Integer.toHexString((b[i]&0xFF) | 0x100).substring(1,3) + " ";
//            if (i % 16 == 15 || i == b.length - 1)
//            {
//                int j;
//                for (j = 16 - i % 16; j > 1; --j)
//                    System.out.print ("   ");
//                System.out.print (" - ");
//                int start = (i / 16) * 16;
//                int end = (b.length < i + 1) ? b.length : (i + 1);
//                for (j = start; j < end; ++j)
//                    if (b[j] >= 32 && b[j] <= 126)
//                        System.out.print ((char)b[j]);
//                    else
//                        System.out.print (".");
//                System.out.println ();
//            }
        }
        //System.out.println();
        
        return hexString;
    }
    
    public static byte[] hexToBytes(String theHex) {

    	theHex = theHex.replaceAll(" ", "");

		byte[] bts = new byte[theHex.length() / 2];
		for (int i = 0; i < bts.length; i++) {
			bts[i] = (byte) Integer.parseInt(theHex.substring(2*i, 2*i+2), 16);
		}
		return bts;
	}
    
    
    public static String decimalToHex(int d){
    	String digits = "0123456789ABCDEF";
        if (d == 0) return "00";
        String hex = "";
        while (d > 0) {
            int digit = d % 16;                // rightmost digit
            hex = digits.charAt(digit) + hex;  // string concatenation
            d = d / 16;
        }
        if (hex.length() == 1){
        	hex = "0" + hex;
        }
        return hex;
    }
    


    
    
    
}

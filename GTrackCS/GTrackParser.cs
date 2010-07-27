using System;
using System.Collections.Generic;

namespace Electroland
{
	public class GTrackParser : ThreadedStringListener
	{

		Dictionary<string, Dictionary<string, float>> curHash = new Dictionary<string, Dictionary<string, float>> ();




		public GTrackParser () : base()
		{
			
		}

		bool isTokenChar(char c) {
			return
				Char.IsLetterOrDigit(c) ||
				c == ',' ||
				c == '.';
			
		}


		string nextWordOrComma (string s, ref int i)
		{
			
			while ((i < s.Length) && (! isTokenChar(s[i++])));
			int start = i-1;
			while ((i < s.Length) && (isTokenChar(s[i++]))) ;
			return s.Substring (start, i - start-1);
			
			
			
		}
		// format
		// id {
		//    name : float;
		//    name : float;
		//    name : float;
		// }
		// ,
		// id {
		//    name : float;
		//    name : float;
		//    name : float;
		// }
		// this could be impvoved by writing our own tokenizer so we do a single pass but this should be fast enough...
		public override void   process (string s)
		{
			
			Dictionary<string, Dictionary<string, float>> futureHash = new Dictionary<string, Dictionary<string, float>> ();
			futureHash.Clear ();
			int index = 0;
			string word = nextWordOrComma (s, ref index);
			while (word != "") {
				// no more words, end of string
				string id = word;
				Dictionary<string, float> dict = new Dictionary<string, float> ();
				word = nextWordOrComma (s, ref index);
				while ((word != ",") && (word != "")) { // white space is required after ,
					// end of object
					string name = word;
					word = nextWordOrComma (s, ref index);
					// value
					dict.Add (name, float.Parse (word));
					word = nextWordOrComma (s, ref index);
				}
				futureHash.Add (id, dict);
				word = nextWordOrComma (s, ref index);
			}
			
			curHash = futureHash;
			
		}


		public Dictionary<string, Dictionary<string, float>> @get ()
		{
			return curHash;
		}
	}
}

//
// Profile.h
// Provides access to INI profile values
//

#ifndef __PROFILE_H__
#define __PROFILE_H__


struct SProfileEntry;

class CProfile
{
public:
  static CProfile *theProfile;
	CProfile();
	~CProfile();

	bool Load(const char* filename);

	const char* String(const char* name, const char* def = 0);
	double Double(const char* name, double def = 0.0);
	float Float(const char* name, float def = 0.0f);
	int Int(const char* name, int def = 0);
	bool Bool(const char* name, bool def = false);
	

protected:
	char* m_strings;
	unsigned int m_offset;
	const char* m_pch;
	SProfileEntry* m_hash[1021];

	void SkipWhitespace();
	const char* GetToken();
	const char* GetString();
	const char* AddString(const char* str, unsigned int len);

	unsigned int Hash(const char* name);
	SProfileEntry* Find(const char *name);
};


#endif // __PROFILE_H__
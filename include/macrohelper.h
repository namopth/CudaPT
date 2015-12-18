#ifndef MACROHELPER_H
#define MACROHELPER_H


#if defined(_DEBUG)
#   define NEW  new//new(_NORMAL_BLOCK,__FILE__, __LINE__)
#else
#	define NEW  new
#endif

#if !defined(DELETE)
#define DELETE(x) if(x) delete x; x=NULL;
#endif

#if !defined(DELETE_SIZE)
#define DELETE_SIZE(x,y) if(x) delete(x,y); x=NULL;
#endif

#if !defined(DELETE_ARRAY)
#define DELETE_ARRAY(x) if (x) delete [] x; x=NULL; 
#endif

#if !defined(RELEASE)
#define RELEASE(x) if(x) x->Release(); x=NULL;
#endif

#if !defined(DEBUG_COUT)
#if _DEBUG
#define DEBUG_COUT(MSG) std::cout << "[" << __FILE__ <<":" << __LINE__ << "] " << MSG << std::endl;
#else
#define DEBUG_COUT(MSG)
#endif
#endif

#endif
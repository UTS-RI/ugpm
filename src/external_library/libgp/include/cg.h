/*
 * cg.h
 *
 *  Created on: Feb 22, 2013
 *      Author: Joao Cunha <joao.cunha@ua.pt>
 */

#ifndef CG_H_
#define CG_H_

#include "gp.h"

namespace libgp
{

class CG
{
public:
	CG();
	virtual ~CG();
	void maximize(GaussianProcess* gp, size_t n=100, bool verbose=1);
};

}

#endif /* CG_H_ */

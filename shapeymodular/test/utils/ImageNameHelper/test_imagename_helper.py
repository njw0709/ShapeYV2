import shapeymodular.utils as utils


class TestImageNameHelper:
    def test_generate_imgnames_from_objname(
        self, test_case_random_obj, test_case_random_axes
    ):
        imgnames = utils.ImageNameHelper.generate_imgnames_from_objname(
            test_case_random_obj
        )
        imgname_with_objname = [
            n for n in utils.SHAPEY200_IMGNAMES if test_case_random_obj in n
        ]
        assert imgnames == imgname_with_objname

        imgnames_with_ax = utils.ImageNameHelper.generate_imgnames_from_objname(
            test_case_random_obj, axes=test_case_random_axes
        )

        imgname_with_axes_true = []
        for n in imgname_with_objname:
            if n.split(".")[0].split("-")[1][0:-2] in test_case_random_axes:
                imgname_with_axes_true.append(n)

        assert imgnames_with_ax == imgname_with_axes_true

    def test_get_objnames_from_imgnames(self, test_case_multiple_sampled_objs):
        objnames = test_case_multiple_sampled_objs
        imgnames = []
        for o in objnames:
            imgnames.extend(utils.ImageNameHelper.generate_imgnames_from_objname(o))
        res_objnames = utils.ImageNameHelper.get_objnames_from_imgnames(imgnames)
        res_objnames = list(set(res_objnames))
        res_objnames.sort()
        assert res_objnames == objnames

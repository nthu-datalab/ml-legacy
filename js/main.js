// Google Analytics
(function (i, s, o, g, r, a, m) {
    i['GoogleAnalyticsObject'] = r;
    i[r] = i[r] || function () {
        (i[r].q = i[r].q || []).push(arguments)
    }, i[r].l = 1 * new Date();
    a = s.createElement(o),
        m = s.getElementsByTagName(o)[0];
    a.async = 1;
    a.src = g;
    m.parentNode.insertBefore(a, m)
})(window, document, 'script', 'https://www.google-analytics.com/analytics.js', 'ga');

ga('create', 'UA-18358274-2', 'auto');
ga('send', 'pageview');

// Send custom GA events
$(document).ready(function () {
    $('a, button').click(function () {
        var id = $(this).attr('id');
        if (id) {
            //alert(id);
            ga('send', 'event', 'shwu-ml', id);
        }
    });
});
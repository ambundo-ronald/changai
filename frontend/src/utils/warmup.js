// (function () {

//     // prevent running twice on same page
//     if (window._changai_warmup_done) return
//     window._changai_warmup_done = true

//     // wait for frappe to finish loading
//     frappe.after_ajax(function () {

//         // fire 20 HTTP requests in parallel
//         Array.from({ length: 20 }, () =>
//             frappe.call({
//                 method: 'changai.changai.api.v2.text2sql_pipeline_v2.load_on_startup',  // hits your Python backend
//                 args: {},
//             })
//         )

//     })
// })()